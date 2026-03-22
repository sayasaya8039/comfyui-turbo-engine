use std::sync::atomic::{AtomicUsize, Ordering};

/// Memory pressure level based on usage ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    /// Usage below 70%.
    Normal,
    /// Usage between 70% and 85%.
    Warning,
    /// Usage between 85% and 95%.
    High,
    /// Usage above 95%.
    Critical,
}

/// Thread-safe memory monitor that tracks allocation budgets.
///
/// All operations are lock-free (atomic). The monitor enforces a hard
/// limit: [`allocate`] returns `false` when the request would exceed
/// `limit_bytes`, and the internal counter stays unchanged.
pub struct MemoryMonitor {
    used_bytes: AtomicUsize,
    limit_bytes: usize,
}

impl MemoryMonitor {
    /// Create a new monitor with the given byte limit.
    pub fn new(limit_bytes: usize) -> Self {
        Self {
            used_bytes: AtomicUsize::new(0),
            limit_bytes,
        }
    }

    /// Try to allocate `bytes`. Returns `true` on success, `false` if the
    /// allocation would exceed the limit (the counter is **not** modified on
    /// failure).
    pub fn allocate(&self, bytes: usize) -> bool {
        loop {
            let current = self.used_bytes.load(Ordering::Relaxed);
            let new = current.saturating_add(bytes);
            if new > self.limit_bytes {
                return false;
            }
            if self
                .used_bytes
                .compare_exchange_weak(current, new, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Release `bytes` back. Saturates at zero (never underflows).
    pub fn deallocate(&self, bytes: usize) {
        loop {
            let current = self.used_bytes.load(Ordering::Relaxed);
            let new = current.saturating_sub(bytes);
            if self
                .used_bytes
                .compare_exchange_weak(current, new, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Currently tracked bytes.
    pub fn used(&self) -> usize {
        self.used_bytes.load(Ordering::Relaxed)
    }

    /// Usage as a fraction in `[0.0, 1.0]`.
    pub fn usage_ratio(&self) -> f64 {
        if self.limit_bytes == 0 {
            return 1.0;
        }
        self.used() as f64 / self.limit_bytes as f64
    }

    /// Current memory pressure level.
    pub fn pressure(&self) -> MemoryPressure {
        let ratio = self.usage_ratio();
        if ratio > 0.95 {
            MemoryPressure::Critical
        } else if ratio > 0.85 {
            MemoryPressure::High
        } else if ratio > 0.70 {
            MemoryPressure::Warning
        } else {
            MemoryPressure::Normal
        }
    }

    /// Reset the counter to zero.
    pub fn reset(&self) {
        self.used_bytes.store(0, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_alloc() {
        let mon = MemoryMonitor::new(1000);
        assert!(mon.allocate(500));
        assert_eq!(mon.used(), 500);
        assert!(mon.allocate(300));
        assert_eq!(mon.used(), 800);
    }

    #[test]
    fn test_pressure_levels() {
        let mon = MemoryMonitor::new(1000);

        // Normal (<70%)
        mon.allocate(600);
        assert_eq!(mon.pressure(), MemoryPressure::Normal);

        // Warning (70-85%)
        mon.reset();
        mon.allocate(750);
        assert_eq!(mon.pressure(), MemoryPressure::Warning);

        // High (85-95%)
        mon.reset();
        mon.allocate(900);
        assert_eq!(mon.pressure(), MemoryPressure::High);

        // Critical (>95%)
        mon.reset();
        mon.allocate(960);
        assert_eq!(mon.pressure(), MemoryPressure::Critical);
    }

    #[test]
    fn test_allocation_limit_enforcement() {
        let mon = MemoryMonitor::new(1000);
        assert!(mon.allocate(800));
        // This would exceed the limit (800 + 300 = 1100 > 1000)
        assert!(!mon.allocate(300));
        // Counter should remain at 800
        assert_eq!(mon.used(), 800);
    }

    #[test]
    fn test_deallocate() {
        let mon = MemoryMonitor::new(1000);
        mon.allocate(500);
        mon.deallocate(200);
        assert_eq!(mon.used(), 300);

        // Deallocate more than used — saturates at 0
        mon.deallocate(9999);
        assert_eq!(mon.used(), 0);
    }

    #[test]
    fn test_reset() {
        let mon = MemoryMonitor::new(1000);
        mon.allocate(750);
        assert_eq!(mon.used(), 750);
        mon.reset();
        assert_eq!(mon.used(), 0);
        assert_eq!(mon.pressure(), MemoryPressure::Normal);
    }
}
