"""Checks what is present/missing for compiling llama-cpp-python with Vulkan on Windows.
Run it from the SAME terminal you use for pip install (ideally the x64 Native Tools Command Prompt for VS 2022):
    python check_build_env.py
"""
import os
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path

OK, BAD, WARN = "[ OK ]", "[MISSING]", "[ WARN ]"
problems = []


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, shell=True, timeout=20)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return str(e)


def report(name, ok, detail="", fix="", warn=False):
    tag = OK if ok else (WARN if warn else BAD)
    print(f"{tag} {name}" + (f"  ->  {detail}" if detail else ""))
    if not ok:
        problems.append((name, fix))


# Python
bits = struct.calcsize("P") * 8
report("Python 64-bit", bits == 64, f"{sys.version.split()[0]} ({bits}-bit)", "Install 64-bit Python")
v = sys.version_info
report("Python version 3.10-3.12", (3, 10) <= (v.major, v.minor) <= (3, 12),
       f"{v.major}.{v.minor}", "Use Python 3.11 or 3.12", warn=True)

# MSVC compiler
cl = shutil.which("cl")
report("MSVC compiler (cl.exe) on PATH", bool(cl), cl or "",
       "Open 'x64 Native Tools Command Prompt for VS 2022' (not normal PowerShell/cmd), "
       "or install VS Build Tools 2022 with 'Desktop development with C++'")

# Visual Studio install
vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
    "Microsoft Visual Studio/Installer/vswhere.exe"
if vswhere.exists():
    out = run(f'"{vswhere}" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath')
    report("Visual Studio C++ tools installed", bool(out.strip()), out.strip(),
           "In VS Installer, tick 'Desktop development with C++'")
else:
    report("Visual Studio / Build Tools installed", False, "",
           "Install Visual Studio Build Tools 2022 -> 'Desktop development with C++'")

# Windows SDK
sdk_dir = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits/10/Include"
report("Windows 10/11 SDK", sdk_dir.exists() and any(sdk_dir.iterdir()) if sdk_dir.exists() else False,
       str(sdk_dir), "Tick 'Windows 11 SDK' in VS Build Tools installer")

# CMake / Ninja
for tool, fix in (("cmake", "pip install cmake  (or install from cmake.org)"),
                  ("ninja", "pip install ninja")):
    p = shutil.which(tool)
    ver = run(f"{tool} --version").splitlines()[0] if p else ""
    report(tool, bool(p), f"{p}  {ver}" if p else "", fix)

# Vulkan SDK
vk = os.environ.get("VULKAN_SDK")
report("VULKAN_SDK env variable", bool(vk), vk or "",
       "Install LunarG Vulkan SDK (vulkan.lunarg.com), then RESTART the terminal")
if vk:
    report("Vulkan headers", (Path(vk) / "Include/vulkan/vulkan.h").exists(), "",
           "Reinstall Vulkan SDK")
    report("Vulkan library (vulkan-1.lib)", (Path(vk) / "Lib/vulkan-1.lib").exists(), "",
           "Reinstall Vulkan SDK")
g = shutil.which("glslc")
report("glslc (shader compiler)", bool(g), g or "",
       "Reinstall Vulkan SDK and keep 'Shader Toolchain' ticked; add %VULKAN_SDK%\\Bin to PATH")

# GPU Vulkan runtime
vi = shutil.which("vulkaninfo")
if vi:
    out = run("vulkaninfo --summary")
    gpus = [l.strip() for l in out.splitlines() if "deviceName" in l]
    report("Vulkan GPU detected", bool(gpus), "; ".join(gpus), "Update AMD GPU driver")
else:
    report("vulkaninfo", False, "", "Comes with the Vulkan SDK; also update AMD driver", warn=True)

# Wrong compiler hijacking the build (conda MinGW / gcc)
for t in ("gcc", "g++", "clang"):
    p = shutil.which(t)
    if p:
        report(f"{t} also on PATH (can hijack the build)", False, p,
               f"Set CC=cl and CXX=cl, or remove it from PATH", warn=True)

# Env vars that influence the build
for k in ("CC", "CXX", "CMAKE_ARGS", "CMAKE_GENERATOR", "FORCE_CMAKE"):
    print(f"[ INFO ] {k} = {os.environ.get(k)}")

print("\n" + "=" * 60)
if problems:
    print("Things to fix:")
    for n, f in problems:
        print(f"  - {n}: {f}")
else:
    print("Everything needed looks present. If it still fails, run pip with --verbose "
          "and send the lines before 'Configuring incomplete'.")
print(f"\nSystem: {platform.platform()}")