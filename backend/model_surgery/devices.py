"""Target-device profiles for memory-fit estimates.

Capacities are nominal hardware figures; reserved/overhead values are
conservative planning assumptions, not measurements. Every profile lists its
assumptions so the UI can show them next to the verdict.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GiB = 1024**3
MiB = 1024**2


class DeviceProfile(BaseModel):
    id: str
    name: str
    memory_kind: Literal["system_ram", "discrete_gpu", "unified"]
    capacity_bytes: int = Field(gt=0)
    reserved_bytes: int = Field(ge=0)  # OS, desktop, other apps
    runtime_overhead_bytes: int = Field(ge=0)  # runtime/driver context (e.g. CUDA)
    host_ram_bytes: int | None = Field(default=None, ge=0)  # for discrete GPU partial offload
    bandwidth_gbps: float | None = Field(default=None, gt=0)
    assumptions: list[str] = Field(default_factory=list)

    @property
    def usable_bytes(self) -> int:
        return max(0, self.capacity_bytes - self.reserved_bytes)


DEFAULT_PROFILES: list[DeviceProfile] = [
    DeviceProfile(
        id="laptop-cpu-16gb",
        name="Laptop · CPU inference · 16 GB RAM",
        memory_kind="system_ram",
        capacity_bytes=16 * GiB,
        reserved_bytes=5 * GiB,
        runtime_overhead_bytes=200 * MiB,
        bandwidth_gbps=60.0,
        assumptions=[
            "16 GB system RAM; ~5 GB reserved for Windows 11, browser and background apps",
            "~200 MB llama.cpp CPU runtime overhead",
            "~60 GB/s dual-channel DDR5 bandwidth (laptop-typical, varies widely)",
        ],
    ),
    DeviceProfile(
        id="laptop-gpu-8gb",
        name="Laptop · 8 GB VRAM GPU + 16 GB RAM",
        memory_kind="discrete_gpu",
        capacity_bytes=8 * GiB,
        reserved_bytes=512 * MiB,
        runtime_overhead_bytes=450 * MiB,
        host_ram_bytes=11 * GiB,
        bandwidth_gbps=256.0,
        assumptions=[
            "8 GB dedicated VRAM; ~0.5 GB reserved for the display/compositor",
            "~450 MB CUDA context and cuBLAS workspace",
            "layers that do not fit are assumed to stay in system RAM (partial offload)",
            "~256 GB/s VRAM bandwidth (mid-range laptop GPU)",
        ],
    ),
    DeviceProfile(
        id="jetson-orin-nano-8gb",
        name="Jetson Orin Nano 8GB · desktop",
        memory_kind="unified",
        capacity_bytes=int(7.4 * GiB),
        reserved_bytes=int(1.8 * GiB),
        runtime_overhead_bytes=350 * MiB,
        bandwidth_gbps=68.0,
        assumptions=[
            "8 GB LPDDR5 shared by CPU and GPU; ~7.4 GiB visible to Linux after carve-outs",
            "~1.8 GiB reserved for JetPack Ubuntu desktop (GNOME) and system services",
            "~350 MB CUDA context; GPU allocations come from the same pool as the CPU",
            "68 GB/s memory bandwidth (102 GB/s in JetPack 6.2 'Super' MAXN mode)",
        ],
    ),
    DeviceProfile(
        id="jetson-orin-nano-8gb-headless",
        name="Jetson Orin Nano 8GB · headless",
        memory_kind="unified",
        capacity_bytes=int(7.4 * GiB),
        reserved_bytes=int(0.8 * GiB),
        runtime_overhead_bytes=350 * MiB,
        bandwidth_gbps=68.0,
        assumptions=[
            "same hardware as the desktop profile with the GUI disabled (multi-user.target)",
            "~0.8 GiB reserved for the kernel and system services",
            "68 GB/s memory bandwidth (102 GB/s in JetPack 6.2 'Super' MAXN mode)",
        ],
    ),
]


def get_profile(profile_id: str) -> DeviceProfile | None:
    return next((p for p in DEFAULT_PROFILES if p.id == profile_id), None)
