use std::collections::{HashSet, VecDeque};

use crate::workflow::{InputValue, Workflow};

/// Remove nodes that cannot reach any of the given output nodes.
///
/// Performs a **backward BFS** from `output_node_ids`, following
/// [`InputValue::Link`] edges in reverse. Any node not visited is dead
/// and gets removed from the returned workflow.
pub fn remove_dead_nodes(workflow: &Workflow, output_node_ids: &[&str]) -> Workflow {
    let reachable = reachable_set(workflow, output_node_ids);

    let nodes = workflow
        .nodes
        .iter()
        .filter(|(id, _)| reachable.contains(id.as_str()))
        .map(|(id, node)| (id.clone(), node.clone()))
        .collect();

    Workflow { nodes }
}

/// Count how many nodes in the workflow are dead (unreachable from outputs).
pub fn count_dead_nodes(workflow: &Workflow, output_node_ids: &[&str]) -> usize {
    let reachable = reachable_set(workflow, output_node_ids);
    workflow.nodes.len() - reachable.len()
}

/// BFS backward from output nodes, returning the set of reachable node IDs.
fn reachable_set<'a>(workflow: &'a Workflow, output_node_ids: &[&'a str]) -> HashSet<&'a str> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    for &out_id in output_node_ids {
        if workflow.nodes.contains_key(out_id) {
            visited.insert(out_id);
            queue.push_back(out_id);
        }
    }

    while let Some(node_id) = queue.pop_front() {
        if let Some(node) = workflow.nodes.get(node_id) {
            for input in node.inputs.values() {
                if let InputValue::Link(source_id, _) = input {
                    let src: &str = source_id.as_str();
                    if workflow.nodes.contains_key(src) && !visited.contains(src) {
                        visited.insert(
                            // Borrow from the HashMap key so the lifetime is 'a.
                            workflow
                                .nodes
                                .get_key_value(src)
                                .map(|(k, _)| k.as_str())
                                .unwrap(),
                        );
                        queue.push_back(
                            workflow
                                .nodes
                                .get_key_value(src)
                                .map(|(k, _)| k.as_str())
                                .unwrap(),
                        );
                    }
                }
            }
        }
    }

    visited
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::Workflow;

    /// Node 3 is disconnected — should be removed.
    #[test]
    fn test_remove_dead_node() {
        let json = r#"{
            "1": { "class_type": "CheckpointLoader", "inputs": {} },
            "2": { "class_type": "KSampler", "inputs": { "model": ["1", 0] } },
            "3": { "class_type": "Orphan", "inputs": {} }
        }"#;
        let wf = Workflow::from_json(json).unwrap();
        let cleaned = remove_dead_nodes(&wf, &["2"]);

        assert_eq!(cleaned.nodes.len(), 2);
        assert!(cleaned.nodes.contains_key("1"));
        assert!(cleaned.nodes.contains_key("2"));
        assert!(!cleaned.nodes.contains_key("3"));

        assert_eq!(count_dead_nodes(&wf, &["2"]), 1);
    }

    /// All nodes are reachable — nothing removed.
    #[test]
    fn test_no_dead_nodes() {
        let json = r#"{
            "1": { "class_type": "Source", "inputs": {} },
            "2": { "class_type": "Sink", "inputs": { "data": ["1", 0] } }
        }"#;
        let wf = Workflow::from_json(json).unwrap();
        let cleaned = remove_dead_nodes(&wf, &["2"]);

        assert_eq!(cleaned.nodes.len(), 2);
        assert_eq!(count_dead_nodes(&wf, &["2"]), 0);
    }

    /// No output nodes specified — everything is dead.
    #[test]
    fn test_all_dead_no_outputs() {
        let json = r#"{
            "1": { "class_type": "A", "inputs": {} },
            "2": { "class_type": "B", "inputs": {} }
        }"#;
        let wf = Workflow::from_json(json).unwrap();
        let cleaned = remove_dead_nodes(&wf, &[]);

        assert_eq!(cleaned.nodes.len(), 0);
        assert_eq!(count_dead_nodes(&wf, &[]), 2);
    }
}
