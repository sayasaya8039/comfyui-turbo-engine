use std::collections::HashMap;
use std::sync::Mutex;

use crate::dag::Dag;
use crate::device::Device;
use crate::error::{ComfyError, ComfyResult};
use crate::node::{NodeInputs, NodeOutputs, NodeValue};
use crate::registry::NodeRegistry;
use crate::workflow::{InputValue, Workflow};

/// Statistics about scheduler execution.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub total_nodes: usize,
    pub gpu_nodes: usize,
    pub npu_nodes: usize,
    pub cpu_nodes: usize,
}

/// A depth-based scheduler that groups independent nodes for parallel execution
/// and tracks device placement via `Node::device_hint()`.
pub struct DeviceScheduler {
    registry: NodeRegistry,
    stats: Mutex<SchedulerStats>,
}

impl DeviceScheduler {
    pub fn new(registry: NodeRegistry) -> Self {
        Self {
            registry,
            stats: Mutex::new(SchedulerStats::default()),
        }
    }

    /// Execute a workflow JSON string using depth-based scheduling.
    ///
    /// 1. Parses workflow JSON into a DAG
    /// 2. Computes depth for each node (longest path from a root)
    /// 3. Groups nodes by depth level
    /// 4. Executes each depth group in order (nodes within a group are independent)
    /// 5. Tracks device assignment statistics
    pub fn execute(&self, json: &str) -> ComfyResult<HashMap<String, NodeOutputs>> {
        let workflow = Workflow::from_json(json)?;
        let dag = Dag::from_workflow(&workflow)?;
        let sorted = dag.topological_sort()?;

        // Compute depth for each node
        let depths = Self::compute_depths(&sorted, &dag);

        // Group nodes by depth
        let max_depth = depths.values().copied().max().unwrap_or(0);
        let mut depth_groups: Vec<Vec<String>> = vec![Vec::new(); max_depth + 1];
        for node_id in &sorted {
            let d = depths[node_id];
            depth_groups[d].push(node_id.clone());
        }

        let mut all_outputs: HashMap<String, NodeOutputs> = HashMap::new();
        let mut stats = SchedulerStats::default();

        // Execute each depth group
        for group in &depth_groups {
            // For now, iterate sequentially within a group.
            // (Parallel execution via rayon can be added when real GPU tasks exist.)
            for node_id in group {
                let workflow_node = workflow.nodes.get(node_id).ok_or_else(|| {
                    ComfyError::NodeNotFound(node_id.clone())
                })?;

                let node_impl =
                    self.registry.create(&workflow_node.class_type).ok_or_else(|| {
                        ComfyError::NodeNotFound(format!(
                            "class_type '{}' not registered",
                            workflow_node.class_type
                        ))
                    })?;

                // Track device hint
                let device = node_impl.device_hint();
                stats.total_nodes += 1;
                match device {
                    Device::Gpu(_) => stats.gpu_nodes += 1,
                    Device::Npu(_) => stats.npu_nodes += 1,
                    Device::Cpu => stats.cpu_nodes += 1,
                }

                // Build inputs
                let mut inputs = NodeInputs::new();
                for (input_name, input_val) in &workflow_node.inputs {
                    let resolved = Self::resolve_input(input_val, &all_outputs)?;
                    inputs.set(input_name.clone(), resolved);
                }

                // Execute
                let outputs = node_impl.execute(&inputs).map_err(|e| {
                    ComfyError::ExecutionError {
                        node: node_id.clone(),
                        message: e.to_string(),
                    }
                })?;

                all_outputs.insert(node_id.clone(), outputs);
            }
        }

        // Store stats
        if let Ok(mut s) = self.stats.lock() {
            *s = stats;
        }

        Ok(all_outputs)
    }

    /// Get a snapshot of the latest execution stats.
    pub fn stats(&self) -> SchedulerStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Compute the depth of each node (longest path from any root).
    fn compute_depths(sorted: &[String], dag: &Dag) -> HashMap<String, usize> {
        let mut depths: HashMap<String, usize> = HashMap::new();

        for node_id in sorted {
            let deps = dag.dependencies(node_id);
            let depth = if deps.is_empty() {
                0
            } else {
                deps.iter()
                    .filter_map(|dep| depths.get(dep))
                    .max()
                    .copied()
                    .unwrap_or(0)
                    + 1
            };
            depths.insert(node_id.clone(), depth);
        }

        depths
    }

    /// Resolve an input value, following links to previously computed outputs.
    fn resolve_input(
        input: &InputValue,
        all_outputs: &HashMap<String, NodeOutputs>,
    ) -> ComfyResult<NodeValue> {
        match input {
            InputValue::Link(source_id, output_index) => {
                let source_outputs = all_outputs.get(source_id).ok_or_else(|| {
                    ComfyError::ExecutionError {
                        node: source_id.clone(),
                        message: "output not yet computed".to_string(),
                    }
                })?;
                let key = format!("output_{output_index}");
                source_outputs
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| ComfyError::ExecutionError {
                        node: source_id.clone(),
                        message: format!("no output at index {output_index}"),
                    })
            }
            InputValue::Float(v) => Ok(NodeValue::Float(*v)),
            InputValue::Int(v) => Ok(NodeValue::Int(*v)),
            InputValue::String(s) => Ok(NodeValue::String(s.clone())),
            InputValue::Bool(b) => Ok(NodeValue::Bool(*b)),
            InputValue::None => Ok(NodeValue::None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Node, NodeMetadata};

    /// A CPU-only node that passes through its float "value" input.
    struct CpuNode;

    impl Node for CpuNode {
        fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let value = inputs.get_float("value")?;
            let mut outputs = NodeOutputs::new();
            outputs.set("output_0", NodeValue::Float(value));
            Ok(outputs)
        }

        fn metadata(&self) -> NodeMetadata {
            NodeMetadata {
                name: "CpuNode".into(),
                display_name: "CPU Node".into(),
                category: "test".into(),
                description: "A CPU test node".into(),
                output_node: false,
            }
        }

        fn device_hint(&self) -> Device {
            Device::Cpu
        }
    }

    /// A GPU-preferring node that doubles its float "x" input.
    struct GpuNode;

    impl Node for GpuNode {
        fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let x = inputs.get_float("x")?;
            let mut outputs = NodeOutputs::new();
            outputs.set("output_0", NodeValue::Float(x * 2.0));
            Ok(outputs)
        }

        fn metadata(&self) -> NodeMetadata {
            NodeMetadata {
                name: "GpuNode".into(),
                display_name: "GPU Node".into(),
                category: "test".into(),
                description: "A GPU test node".into(),
                output_node: false,
            }
        }

        fn device_hint(&self) -> Device {
            Device::Gpu(0)
        }
    }

    /// A node that adds two float inputs "a" and "b".
    struct AddNode;

    impl Node for AddNode {
        fn execute(&self, inputs: &NodeInputs) -> ComfyResult<NodeOutputs> {
            let a = inputs.get_float("a")?;
            let b = inputs.get_float("b")?;
            let mut outputs = NodeOutputs::new();
            outputs.set("output_0", NodeValue::Float(a + b));
            Ok(outputs)
        }

        fn metadata(&self) -> NodeMetadata {
            NodeMetadata {
                name: "AddNode".into(),
                display_name: "Add".into(),
                category: "test".into(),
                description: "Adds two values".into(),
                output_node: true,
            }
        }
    }

    fn build_registry() -> NodeRegistry {
        let mut reg = NodeRegistry::new();
        reg.register("CpuNode", || Box::new(CpuNode));
        reg.register("GpuNode", || Box::new(GpuNode));
        reg.register("AddNode", || Box::new(AddNode));
        reg
    }

    #[test]
    fn test_scheduler_routes_to_correct_queue() {
        // CpuNode(value=5.0) -> GpuNode(x=link) -> result should be 10.0
        let json = r#"{
            "1": { "class_type": "CpuNode", "inputs": { "value": 5.0 } },
            "2": { "class_type": "GpuNode", "inputs": { "x": ["1", 0] } }
        }"#;

        let scheduler = DeviceScheduler::new(build_registry());
        let results = scheduler.execute(json).unwrap();

        // CpuNode outputs 5.0
        match results["1"].get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 5.0),
            other => panic!("Expected Float(5.0), got {other:?}"),
        }

        // GpuNode doubles it to 10.0
        match results["2"].get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 10.0),
            other => panic!("Expected Float(10.0), got {other:?}"),
        }

        let stats = scheduler.stats();
        assert_eq!(stats.cpu_nodes, 1);
        assert_eq!(stats.gpu_nodes, 1);
    }

    #[test]
    fn test_scheduler_independent_nodes_parallel() {
        // Two independent CpuNodes at depth 0 — both should complete.
        let json = r#"{
            "a": { "class_type": "CpuNode", "inputs": { "value": 3.0 } },
            "b": { "class_type": "CpuNode", "inputs": { "value": 7.0 } }
        }"#;

        let scheduler = DeviceScheduler::new(build_registry());
        let results = scheduler.execute(json).unwrap();

        assert_eq!(results.len(), 2);
        match results["a"].get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 3.0),
            other => panic!("Expected Float(3.0), got {other:?}"),
        }
        match results["b"].get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 7.0),
            other => panic!("Expected Float(7.0), got {other:?}"),
        }
    }

    #[test]
    fn test_scheduler_respects_dependencies() {
        // Chain of 3: CpuNode(1.0) -> GpuNode(x2) -> GpuNode(x2) = 4.0
        let json = r#"{
            "1": { "class_type": "CpuNode", "inputs": { "value": 1.0 } },
            "2": { "class_type": "GpuNode", "inputs": { "x": ["1", 0] } },
            "3": { "class_type": "GpuNode", "inputs": { "x": ["2", 0] } }
        }"#;

        let scheduler = DeviceScheduler::new(build_registry());
        let results = scheduler.execute(json).unwrap();

        match results["1"].get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 1.0),
            other => panic!("Expected Float(1.0), got {other:?}"),
        }
        match results["2"].get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 2.0),
            other => panic!("Expected Float(2.0), got {other:?}"),
        }
        match results["3"].get("output_0") {
            Some(NodeValue::Float(v)) => assert_eq!(*v, 4.0),
            other => panic!("Expected Float(4.0), got {other:?}"),
        }
    }

    #[test]
    fn test_scheduler_stats() {
        // 2 CpuNodes + 1 GpuNode
        let json = r#"{
            "a": { "class_type": "CpuNode", "inputs": { "value": 1.0 } },
            "b": { "class_type": "CpuNode", "inputs": { "value": 2.0 } },
            "c": { "class_type": "GpuNode", "inputs": { "x": ["a", 0] } }
        }"#;

        let scheduler = DeviceScheduler::new(build_registry());
        let _results = scheduler.execute(json).unwrap();

        let stats = scheduler.stats();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.cpu_nodes, 2);
        assert_eq!(stats.gpu_nodes, 1);
        assert_eq!(stats.npu_nodes, 0);
    }
}
