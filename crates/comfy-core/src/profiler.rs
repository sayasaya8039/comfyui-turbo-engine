use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Summary statistics for a profiled operation.
#[derive(Debug, Clone)]
pub struct ProfileSummary {
    /// Total time spent across all invocations.
    pub total: Duration,
    /// Number of recorded invocations.
    pub count: usize,
    /// Average duration per invocation.
    pub avg: Duration,
    /// Minimum observed duration.
    pub min: Duration,
    /// Maximum observed duration.
    pub max: Duration,
}

/// Thread-safe profiler that records named duration measurements.
///
/// Designed for instrumenting node execution, model loading, and other
/// pipeline stages. All methods are safe to call from multiple threads.
pub struct Profiler {
    entries: Mutex<HashMap<String, Vec<Duration>>>,
}

impl Profiler {
    /// Create a new empty profiler.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Record a duration measurement for the named operation.
    pub fn record(&self, name: &str, duration: Duration) {
        if let Ok(mut entries) = self.entries.lock() {
            entries
                .entry(name.to_string())
                .or_default()
                .push(duration);
        }
    }

    /// Time a closure and record its duration under the given name.
    ///
    /// Returns the closure's return value.
    pub fn time<F, R>(&self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        self.record(name, start.elapsed());
        result
    }

    /// Compute summary statistics for all recorded operations.
    ///
    /// Returns a map from operation name to its `ProfileSummary`.
    pub fn summary(&self) -> HashMap<String, ProfileSummary> {
        let entries = match self.entries.lock() {
            Ok(e) => e,
            Err(_) => return HashMap::new(),
        };

        entries
            .iter()
            .filter_map(|(name, durations)| {
                if durations.is_empty() {
                    return None;
                }
                let total: Duration = durations.iter().sum();
                let count = durations.len();
                let avg = total / count as u32;
                let min = *durations.iter().min().unwrap();
                let max = *durations.iter().max().unwrap();
                Some((
                    name.clone(),
                    ProfileSummary {
                        total,
                        count,
                        avg,
                        min,
                        max,
                    },
                ))
            })
            .collect()
    }

    /// Clear all recorded measurements.
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.clear();
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_summary() {
        let profiler = Profiler::new();

        profiler.record("op_a", Duration::from_millis(100));
        profiler.record("op_a", Duration::from_millis(200));
        profiler.record("op_a", Duration::from_millis(300));
        profiler.record("op_b", Duration::from_millis(50));

        let summary = profiler.summary();
        assert_eq!(summary.len(), 2);

        let a = summary.get("op_a").expect("op_a should exist");
        assert_eq!(a.count, 3);
        assert_eq!(a.total, Duration::from_millis(600));
        assert_eq!(a.avg, Duration::from_millis(200));
        assert_eq!(a.min, Duration::from_millis(100));
        assert_eq!(a.max, Duration::from_millis(300));

        let b = summary.get("op_b").expect("op_b should exist");
        assert_eq!(b.count, 1);
        assert_eq!(b.total, Duration::from_millis(50));
    }

    #[test]
    fn test_time_closure() {
        let profiler = Profiler::new();

        let result = profiler.time("compute", || {
            // Simulate some work
            let mut sum = 0u64;
            for i in 0..1000 {
                sum += i;
            }
            sum
        });

        assert_eq!(result, 499_500);

        let summary = profiler.summary();
        let entry = summary.get("compute").expect("compute should exist");
        assert_eq!(entry.count, 1);
        assert!(entry.total > Duration::ZERO);
    }

    #[test]
    fn test_clear() {
        let profiler = Profiler::new();

        profiler.record("x", Duration::from_millis(10));
        profiler.record("y", Duration::from_millis(20));
        assert_eq!(profiler.summary().len(), 2);

        profiler.clear();
        assert!(profiler.summary().is_empty());
    }
}
