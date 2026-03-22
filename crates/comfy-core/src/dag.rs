use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{ComfyError, ComfyResult};
use crate::workflow::{InputValue, Workflow};

/// A directed acyclic graph built from a workflow.
#[derive(Debug, Clone)]
pub struct Dag {
    /// All node IDs in the graph.
    pub node_ids: Vec<String>,
    /// Edges: (from_node_id, to_node_id, output_index).
    pub edges: Vec<(String, String, usize)>,
    /// Forward adjacency: node_id -> list of (target_node_id, output_index).
    pub forward: HashMap<String, Vec<(String, usize)>>,
    /// Reverse adjacency: node_id -> list of (source_node_id, output_index).
    pub reverse: HashMap<String, Vec<(String, usize)>>,
    /// In-degree for each node.
    pub in_degree: HashMap<String, usize>,
}

impl Dag {
    /// Build a DAG from a parsed workflow by extracting link dependencies.
    pub fn from_workflow(workflow: &Workflow) -> ComfyResult<Self> {
        let node_ids: Vec<String> = {
            let mut ids: Vec<String> = workflow.nodes.keys().cloned().collect();
            ids.sort();
            ids
        };

        let mut edges = Vec::new();
        let mut forward: HashMap<String, Vec<(String, usize)>> = HashMap::new();
        let mut reverse: HashMap<String, Vec<(String, usize)>> = HashMap::new();
        let mut in_degree: HashMap<String, usize> = HashMap::new();

        // Initialize all nodes
        for id in &node_ids {
            forward.entry(id.clone()).or_default();
            reverse.entry(id.clone()).or_default();
            in_degree.entry(id.clone()).or_insert(0);
        }

        // Extract edges from input links
        for (node_id, node) in &workflow.nodes {
            for (_input_name, input_val) in &node.inputs {
                if let InputValue::Link(source_id, output_index) = input_val {
                    if !workflow.nodes.contains_key(source_id) {
                        return Err(ComfyError::NodeNotFound(source_id.clone()));
                    }
                    edges.push((source_id.clone(), node_id.clone(), *output_index));
                    forward
                        .entry(source_id.clone())
                        .or_default()
                        .push((node_id.clone(), *output_index));
                    reverse
                        .entry(node_id.clone())
                        .or_default()
                        .push((source_id.clone(), *output_index));
                    *in_degree.entry(node_id.clone()).or_insert(0) += 1;
                }
            }
        }

        Ok(Dag {
            node_ids,
            edges,
            forward,
            reverse,
            in_degree,
        })
    }

    /// Topological sort using Kahn's algorithm.
    /// Returns an error if the graph contains a cycle.
    pub fn topological_sort(&self) -> ComfyResult<Vec<String>> {
        let mut in_degree = self.in_degree.clone();
        let mut queue: VecDeque<String> = VecDeque::new();

        // Start with nodes that have no incoming edges
        let mut zero_degree: Vec<&String> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(id, _)| id)
            .collect();
        zero_degree.sort(); // deterministic ordering
        for id in zero_degree {
            queue.push_back(id.clone());
        }

        let mut sorted = Vec::with_capacity(self.node_ids.len());

        while let Some(node_id) = queue.pop_front() {
            sorted.push(node_id.clone());

            if let Some(neighbors) = self.forward.get(&node_id) {
                // Collect and sort for determinism
                let mut targets: Vec<&String> = neighbors.iter().map(|(id, _)| id).collect();
                targets.sort();
                targets.dedup();

                for target in targets {
                    if let Some(deg) = in_degree.get_mut(target) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(target.clone());
                        }
                    }
                }
            }
        }

        if sorted.len() != self.node_ids.len() {
            return Err(ComfyError::CycleDetected);
        }

        Ok(sorted)
    }

    /// Returns the direct dependencies (predecessors) of a node.
    pub fn dependencies(&self, node_id: &str) -> Vec<String> {
        self.reverse
            .get(node_id)
            .map(|deps| {
                let mut ids: Vec<String> = deps.iter().map(|(id, _)| id.clone()).collect();
                ids.sort();
                ids.dedup();
                ids
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::Workflow;

    #[test]
    fn test_topo_sort_linear() {
        // A -> B -> C
        let json = r#"{
            "A": { "class_type": "Source", "inputs": {} },
            "B": { "class_type": "Process", "inputs": { "x": ["A", 0] } },
            "C": { "class_type": "Output", "inputs": { "x": ["B", 0] } }
        }"#;

        let workflow = Workflow::from_json(json).unwrap();
        let dag = Dag::from_workflow(&workflow).unwrap();
        let sorted = dag.topological_sort().unwrap();

        let pos_a = sorted.iter().position(|x| x == "A").unwrap();
        let pos_b = sorted.iter().position(|x| x == "B").unwrap();
        let pos_c = sorted.iter().position(|x| x == "C").unwrap();

        assert!(pos_a < pos_b, "A must come before B");
        assert!(pos_b < pos_c, "B must come before C");
    }

    #[test]
    fn test_topo_sort_diamond() {
        // A -> B, A -> C, B -> D, C -> D
        let json = r#"{
            "A": { "class_type": "Source", "inputs": {} },
            "B": { "class_type": "Left", "inputs": { "x": ["A", 0] } },
            "C": { "class_type": "Right", "inputs": { "x": ["A", 0] } },
            "D": { "class_type": "Merge", "inputs": { "left": ["B", 0], "right": ["C", 0] } }
        }"#;

        let workflow = Workflow::from_json(json).unwrap();
        let dag = Dag::from_workflow(&workflow).unwrap();
        let sorted = dag.topological_sort().unwrap();

        assert_eq!(sorted.len(), 4);
        let pos_a = sorted.iter().position(|x| x == "A").unwrap();
        let pos_b = sorted.iter().position(|x| x == "B").unwrap();
        let pos_c = sorted.iter().position(|x| x == "C").unwrap();
        let pos_d = sorted.iter().position(|x| x == "D").unwrap();

        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn test_cycle_detection() {
        // A -> B -> A (cycle) - we need to manually construct this
        // since our parser would parse it fine but the cycle is in links
        let json = r#"{
            "A": { "class_type": "NodeA", "inputs": { "x": ["B", 0] } },
            "B": { "class_type": "NodeB", "inputs": { "x": ["A", 0] } }
        }"#;

        let workflow = Workflow::from_json(json).unwrap();
        let dag = Dag::from_workflow(&workflow).unwrap();
        let result = dag.topological_sort();

        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("cycle"), "Expected cycle error, got: {err}");
    }

    #[test]
    fn test_dependencies() {
        let json = r#"{
            "A": { "class_type": "Source", "inputs": {} },
            "B": { "class_type": "Source2", "inputs": {} },
            "C": { "class_type": "Merge", "inputs": { "x": ["A", 0], "y": ["B", 0] } }
        }"#;

        let workflow = Workflow::from_json(json).unwrap();
        let dag = Dag::from_workflow(&workflow).unwrap();

        let deps_a = dag.dependencies("A");
        assert!(deps_a.is_empty());

        let deps_c = dag.dependencies("C");
        assert_eq!(deps_c, vec!["A", "B"]);
    }

    #[test]
    fn test_single_node() {
        let json = r#"{
            "1": { "class_type": "Output", "inputs": {} }
        }"#;

        let workflow = Workflow::from_json(json).unwrap();
        let dag = Dag::from_workflow(&workflow).unwrap();
        let sorted = dag.topological_sort().unwrap();
        assert_eq!(sorted, vec!["1"]);
    }

    #[test]
    fn test_missing_link_target() {
        let json = r#"{
            "A": { "class_type": "Node", "inputs": { "x": ["Z", 0] } }
        }"#;

        let workflow = Workflow::from_json(json).unwrap();
        let result = Dag::from_workflow(&workflow);
        assert!(result.is_err());
    }
}
