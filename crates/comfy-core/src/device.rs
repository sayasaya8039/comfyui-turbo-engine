use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a compute device for node execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    Gpu(usize),
    Npu(usize),
    Cpu,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Gpu(i) => write!(f, "GPU:{i}"),
            Device::Npu(i) => write!(f, "NPU:{i}"),
            Device::Cpu => write!(f, "CPU"),
        }
    }
}

/// Capabilities and metadata for a single compute device.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub device: Device,
    pub name: String,
    pub compute_units: u32,
    pub memory_bytes: u64,
    pub supports_f16: bool,
    pub supports_int8: bool,
}

impl DeviceCapabilities {
    /// Returns memory in gigabytes (floating-point).
    pub fn memory_gb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Hardware information gathered from the system.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub devices: Vec<DeviceCapabilities>,
}

/// Detects available hardware devices on the system.
pub struct HardwareDetector;

impl HardwareDetector {
    /// Detect available compute devices.
    ///
    /// Always includes CPU. Checks for NVIDIA GPU via CUDA_PATH
    /// environment variable or standard installation path.
    pub fn detect() -> HardwareInfo {
        let mut devices = Vec::new();

        // Always include CPU
        devices.push(DeviceCapabilities {
            device: Device::Cpu,
            name: "CPU".into(),
            compute_units: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4),
            memory_bytes: 16 * 1024 * 1024 * 1024, // Default 16GB
            supports_f16: false,
            supports_int8: true,
        });

        // Check for NVIDIA GPU
        if std::env::var("CUDA_PATH").is_ok()
            || std::path::Path::new(
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
            )
            .exists()
        {
            devices.push(DeviceCapabilities {
                device: Device::Gpu(0),
                name: "NVIDIA GPU".into(),
                compute_units: 0,
                memory_bytes: 0,
                supports_f16: true,
                supports_int8: true,
            });
        }

        HardwareInfo { devices }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_enum_variants() {
        let gpu = Device::Gpu(0);
        let npu = Device::Npu(1);
        let cpu = Device::Cpu;

        assert_eq!(gpu, Device::Gpu(0));
        assert_eq!(npu, Device::Npu(1));
        assert_eq!(cpu, Device::Cpu);
        assert_ne!(gpu, cpu);
        assert_ne!(gpu, npu);
        assert_ne!(Device::Gpu(0), Device::Gpu(1));
    }

    #[test]
    fn test_device_display() {
        assert_eq!(format!("{}", Device::Gpu(0)), "GPU:0");
        assert_eq!(format!("{}", Device::Gpu(3)), "GPU:3");
        assert_eq!(format!("{}", Device::Npu(1)), "NPU:1");
        assert_eq!(format!("{}", Device::Cpu), "CPU");
    }

    #[test]
    fn test_hardware_detector_returns_at_least_cpu() {
        let info = HardwareDetector::detect();
        assert!(!info.devices.is_empty(), "Must have at least one device");

        let cpu = info
            .devices
            .iter()
            .find(|d| d.device == Device::Cpu);
        assert!(cpu.is_some(), "CPU device must always be present");

        let cpu = cpu.unwrap();
        assert_eq!(cpu.name, "CPU");
        assert!(cpu.compute_units > 0, "CPU must have >0 compute units");
        assert!(cpu.supports_int8);
    }

    #[test]
    fn test_device_capabilities_memory_gb() {
        let cap = DeviceCapabilities {
            device: Device::Gpu(0),
            name: "Test GPU".into(),
            compute_units: 128,
            memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
            supports_f16: true,
            supports_int8: true,
        };
        assert!((cap.memory_gb() - 8.0).abs() < f64::EPSILON);

        let cap_zero = DeviceCapabilities {
            device: Device::Cpu,
            name: "CPU".into(),
            compute_units: 4,
            memory_bytes: 0,
            supports_f16: false,
            supports_int8: true,
        };
        assert!((cap_zero.memory_gb() - 0.0).abs() < f64::EPSILON);
    }
}
