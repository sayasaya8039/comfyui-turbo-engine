use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a compute device for node execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    Gpu(usize),
    Npu(usize),
    Cpu,
}

/// Type of compute task, used for device recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// Heavy model inference (UNet, VAE) - prefers GPU
    ModelInference,
    /// Text encoding (CLIP) - NPU > GPU > CPU
    TextEncoding,
    /// Image processing (resize, crop, etc.) - CPU with rayon
    ImageProcessing,
    /// File I/O operations - CPU
    FileIO,
    /// Small model inference - NPU > GPU > CPU
    SmallModel,
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

impl HardwareInfo {
    /// Recommend the best device for a given task type.
    pub fn recommend_device(&self, task: TaskType) -> Device {
        match task {
            TaskType::ModelInference => {
                // Prefer GPU with most memory
                self.devices
                    .iter()
                    .filter(|d| matches!(d.device, Device::Gpu(_)))
                    .max_by_key(|d| d.memory_bytes)
                    .map(|d| d.device)
                    .unwrap_or(Device::Cpu)
            }
            TaskType::TextEncoding | TaskType::SmallModel => {
                // Prefer NPU > GPU > CPU
                self.devices
                    .iter()
                    .find(|d| matches!(d.device, Device::Npu(_)))
                    .or_else(|| {
                        self.devices
                            .iter()
                            .find(|d| matches!(d.device, Device::Gpu(_)))
                    })
                    .map(|d| d.device)
                    .unwrap_or(Device::Cpu)
            }
            TaskType::ImageProcessing | TaskType::FileIO => Device::Cpu,
        }
    }

    /// Returns the number of CPU cores available.
    pub fn cpu_cores(&self) -> u32 {
        self.devices
            .iter()
            .find(|d| matches!(d.device, Device::Cpu))
            .map(|d| d.compute_units)
            .unwrap_or(4)
    }

    /// Returns total system RAM in bytes.
    pub fn system_ram_bytes(&self) -> u64 {
        self.devices
            .iter()
            .find(|d| matches!(d.device, Device::Cpu))
            .map(|d| d.memory_bytes)
            .unwrap_or(16 * 1024 * 1024 * 1024)
    }

    /// Check if GPU acceleration is available.
    pub fn has_gpu(&self) -> bool {
        self.devices
            .iter()
            .any(|d| matches!(d.device, Device::Gpu(_)))
    }

    /// Check if NPU is available.
    pub fn has_npu(&self) -> bool {
        self.devices
            .iter()
            .any(|d| matches!(d.device, Device::Npu(_)))
    }
}

/// Detects available hardware devices on the system.
pub struct HardwareDetector;

impl HardwareDetector {
    /// Detect available compute devices.
    ///
    /// Always includes CPU with system RAM detection.
    /// Detects NVIDIA GPU via nvidia-smi, falling back to CUDA_PATH check.
    /// Detects Intel NPU via OpenVINO environment or Windows PnP device enumeration.
    pub fn detect() -> HardwareInfo {
        let mut devices = Vec::new();

        // CPU detection
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4);
        let system_ram = Self::detect_system_ram();
        devices.push(DeviceCapabilities {
            device: Device::Cpu,
            name: format!("CPU ({} cores)", cpu_cores),
            compute_units: cpu_cores,
            memory_bytes: system_ram,
            supports_f16: false,
            supports_int8: true,
        });

        // NVIDIA GPU detection via nvidia-smi
        if let Some(gpu) = Self::detect_nvidia_gpu() {
            devices.push(gpu);
        } else if std::env::var("CUDA_PATH").is_ok()
            || std::path::Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").exists()
        {
            devices.push(DeviceCapabilities {
                device: Device::Gpu(0),
                name: "NVIDIA GPU (details unavailable)".into(),
                compute_units: 0,
                memory_bytes: 0,
                supports_f16: true,
                supports_int8: true,
            });
        }

        // Intel NPU detection
        if let Some(npu) = Self::detect_intel_npu() {
            devices.push(npu);
        }

        HardwareInfo { devices }
    }

    /// Detect NVIDIA GPU via nvidia-smi.
    fn detect_nvidia_gpu() -> Option<DeviceCapabilities> {
        let output = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout.lines().next()?;
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 2 {
            return None;
        }

        let name = parts[0].to_string();
        let memory_mb: u64 = parts[1].parse().unwrap_or(0);
        let memory_bytes = memory_mb * 1024 * 1024;

        Some(DeviceCapabilities {
            device: Device::Gpu(0),
            name,
            compute_units: 0, // nvidia-smi doesn't easily expose SM count
            memory_bytes,
            supports_f16: true,
            supports_int8: true,
        })
    }

    /// Detect Intel NPU (AI Boost) or OpenVINO availability.
    fn detect_intel_npu() -> Option<DeviceCapabilities> {
        // Check OpenVINO environment
        if std::env::var("INTEL_OPENVINO_DIR").is_ok() {
            return Some(DeviceCapabilities {
                device: Device::Npu(0),
                name: "Intel NPU (OpenVINO)".into(),
                compute_units: 0,
                memory_bytes: 0,
                supports_f16: true,
                supports_int8: true,
            });
        }

        // Check Windows PnP for Intel AI Boost
        #[cfg(target_os = "windows")]
        {
            let output = std::process::Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    "Get-PnpDevice -Class 'System' -Status 'OK' | Where-Object { $_.FriendlyName -like '*NPU*' -or $_.FriendlyName -like '*AI Boost*' } | Select-Object -First 1 -ExpandProperty FriendlyName",
                ])
                .output()
                .ok()?;

            if output.status.success() {
                let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !name.is_empty() {
                    return Some(DeviceCapabilities {
                        device: Device::Npu(0),
                        name,
                        compute_units: 0,
                        memory_bytes: 0,
                        supports_f16: true,
                        supports_int8: true,
                    });
                }
            }
        }

        None
    }

    /// Detect system RAM.
    fn detect_system_ram() -> u64 {
        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = std::process::Command::new("wmic")
                .args(["os", "get", "TotalVisibleMemorySize", "/value"])
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if let Some(val) = line.strip_prefix("TotalVisibleMemorySize=") {
                        if let Ok(kb) = val.trim().parse::<u64>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
        // Default: 16GB
        16 * 1024 * 1024 * 1024
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
        assert!(cpu.name.starts_with("CPU ("), "CPU name should include core count");
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

    #[test]
    fn test_task_recommendation_with_gpu() {
        let info = HardwareInfo {
            devices: vec![
                DeviceCapabilities {
                    device: Device::Cpu,
                    name: "CPU".into(),
                    compute_units: 8,
                    memory_bytes: 32 * 1024 * 1024 * 1024,
                    supports_f16: false,
                    supports_int8: true,
                },
                DeviceCapabilities {
                    device: Device::Gpu(0),
                    name: "RTX 4090".into(),
                    compute_units: 128,
                    memory_bytes: 24 * 1024 * 1024 * 1024,
                    supports_f16: true,
                    supports_int8: true,
                },
            ],
        };

        assert_eq!(
            info.recommend_device(TaskType::ModelInference),
            Device::Gpu(0)
        );
        assert_eq!(
            info.recommend_device(TaskType::ImageProcessing),
            Device::Cpu
        );
        assert!(info.has_gpu());
        assert!(!info.has_npu());
    }

    #[test]
    fn test_task_recommendation_with_npu() {
        let info = HardwareInfo {
            devices: vec![
                DeviceCapabilities {
                    device: Device::Cpu,
                    name: "CPU".into(),
                    compute_units: 8,
                    memory_bytes: 16 * 1024 * 1024 * 1024,
                    supports_f16: false,
                    supports_int8: true,
                },
                DeviceCapabilities {
                    device: Device::Npu(0),
                    name: "Intel NPU".into(),
                    compute_units: 0,
                    memory_bytes: 0,
                    supports_f16: true,
                    supports_int8: true,
                },
            ],
        };

        assert_eq!(
            info.recommend_device(TaskType::TextEncoding),
            Device::Npu(0)
        );
        assert_eq!(
            info.recommend_device(TaskType::SmallModel),
            Device::Npu(0)
        );
        assert!(info.has_npu());
    }
}
