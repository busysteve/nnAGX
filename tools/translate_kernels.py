#!/usr/bin/env python3
"""Translate nnCL src/kernels.cl into src/kernels.metal.

The numerical kernels stay line for line. Address spaces, ids, barriers, and
atomics change to Metal Shading Language. The float64 sum-of-squares kernels
are dropped: Apple GPUs have no device float64. Fill kernels are appended for
the runtime, which has no clEnqueueFillBuffer.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT.parent / "nnCL" / "src" / "kernels.cl"
OUT = ROOT / "src" / "kernels.metal"

PREAMBLE = r'''// Metal Shading Language kernels for nnAGX.
// Translated from the nnCL OpenCL kernels. One in-order command queue.
// Bfloat16 values are bit patterns. Products widen them and accumulate in float32.
// Fast math is off so divf and the bfloat16 rounding stay exact.
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

int agx_cmpxchg(device int* addr, int comparand, int desired) {
  device atomic_int* object = (device atomic_int*)addr;
  int expected = comparand;
  while (!atomic_compare_exchange_weak_explicit(object, &expected, desired, memory_order_relaxed,
                                                memory_order_relaxed)) {
    if (expected != comparand) return expected;
  }
  return comparand;
}

int agx_atomic_add(device int* addr, int value) {
  return atomic_fetch_add_explicit((device atomic_int*)addr, value, memory_order_relaxed);
}

float block_sum(float value, threadgroup float* buffer, uint t_tid, uint t_tgs) {
  const int tid = (int)t_tid;
  const int width = (int)t_tgs;
  buffer[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int offset = width >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) buffer[tid] += buffer[tid + offset];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float total = buffer[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

float block_max(float value, threadgroup float* buffer, uint t_tid, uint t_tgs) {
  const int tid = (int)t_tid;
  const int width = (int)t_tgs;
  buffer[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int offset = width >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) buffer[tid] = fmax(buffer[tid], buffer[tid + offset]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float total = buffer[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// libclc's reciprocal is an ulp low here (12/3 -> 3.99999976). Normalization
// multiplies that error by rsqrt(eps), which misses a 1e-5 check. One FMA
// residual step restores the correctly rounded quotient.
float divf(float a, float b) {
  const float q = a / b;
  const float r = fma(-q, b, a);
  return fma(r, 1.f / b, q);
}

void atomic_add_f(device float* addr, float val) {
  device int* iaddr = (device int*)addr;
  int expected = agx_cmpxchg(iaddr, 0, 0);
  while (1) {
    union {
      int i;
      float f;
    } next;
    next.f = as_type<float>(expected) + val;
    const int old = agx_cmpxchg(iaddr, expected, next.i);
    if (old == expected) return;
    expected = old;
  }
}

float gelu_forward(float x) {
  const float k_inv_sqrt2 = 0.7071067811865476f;
  return 0.5f * x * (1.f + erf(x * k_inv_sqrt2));
}

float gelu_grad(float x) {
  const float k_inv_sqrt2 = 0.7071067811865476f;
  const float k_inv_sqrt2pi = 0.3989422804014327f;
  const float cdf = 0.5f * (1.f + erf(x * k_inv_sqrt2));
  const float pdf = k_inv_sqrt2pi * exp(-0.5f * x * x);
  return cdf + x * pdf;
}

float sigmoid_stable(float x) {
  if (x >= 0.f) {
    const float z = exp(-x);
    return 1.f / (1.f + z);
  }
  const float z = exp(x);
  return z / (1.f + z);
}

ushort f32_to_bf16(float value) {
  uint bits = as_type<uint>(value);
  uint exp = bits & 0x7f800000u;
  if (exp == 0x7f800000u) return (ushort)(bits >> 16);
  uint lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return (ushort)(bits >> 16);
}

float bf16_to_f32(ushort bits) { return as_type<float>(((uint)bits) << 16); }

'''

FILL = r'''
kernel void fill_f(device float* dst [[buffer(0)]], constant long& n [[buffer(1)]],
                   constant float& value [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if ((long)gid < n) dst[gid] = value;
}

kernel void fill_i(device int* dst [[buffer(0)]], constant long& n [[buffer(1)]],
                   constant int& value [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if ((long)gid < n) dst[gid] = value;
}

kernel void fill_u8(device uchar* dst [[buffer(0)]], constant long& n [[buffer(1)]],
                    constant int& value [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if ((long)gid < n) dst[gid] = (uchar)value;
}
'''

FORBIDDEN = (
    "__global",
    "__local",
    "__kernel",
    "__constant",
    "get_global_id",
    "get_local_id",
    "get_local_size",
    "get_group_id",
    "get_num_groups",
    "CLK_",
    "atomic_cmpxchg",
    "as_float",
    "as_uint",
    "as_int",
    "HAVE_FP64",
)


def strip_ifdef(text, macro):
    out = []
    skipping = False
    depth = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not skipping and stripped == f"#ifdef {macro}":
            skipping = True
            depth = 1
            continue
        if skipping:
            if stripped.startswith("#ifdef") or stripped.startswith("#if"):
                depth += 1
            elif stripped.startswith("#endif"):
                depth -= 1
                if depth == 0:
                    skipping = False
            continue
        out.append(line)
    if skipping:
        raise SystemExit(f"unclosed ifdef {macro}")
    return "".join(out)


def match_paren(text, open_at):
    depth = 0
    i = open_at
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise SystemExit("unbalanced paren")


def match_brace(text, open_at):
    depth = 0
    i = open_at
    while i < len(text):
        c = text[i]
        if c == "/" and i + 1 < len(text) and text[i + 1] == "/":
            i = text.find("\n", i)
            if i < 0:
                break
            continue
        if c == "/" and i + 1 < len(text) and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
            continue
        if c == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                if text[i] == "\\":
                    i += 1
                i += 1
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise SystemExit("unbalanced brace")


def split_params(param_str):
    parts = []
    cur = []
    depth = 0
    for ch in param_str:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def split_type_name(param):
    param = " ".join(param.replace("\n", " ").split())
    match = re.match(r"^(.*\S)\s+([A-Za-z_][A-Za-z0-9_]*)$", param)
    if not match:
        raise SystemExit(f"bad parameter: {param}")
    return match.group(1).strip(), match.group(2)


def translate_body(body, is_3d):
    if "get_local_id(1)" in body or "get_global_id(3)" in body:
        raise SystemExit("unsupported id axis")
    if "barrier(" in body and "CLK_LOCAL_MEM_FENCE" not in body and "CLK_GLOBAL_MEM_FENCE" not in body:
        raise SystemExit("unsupported barrier")
    out = body
    out = out.replace("const __global", "const device")
    out = out.replace("__global const", "const device")
    out = out.replace("__global", "device")
    out = out.replace("__local", "threadgroup")
    if is_3d:
        out = out.replace("get_global_id(2)", "gid.z")
        out = out.replace("get_global_id(1)", "gid.y")
        out = out.replace("get_global_id(0)", "gid.x")
    else:
        if "get_global_id(1)" in out or "get_global_id(2)" in out:
            raise SystemExit("1d kernel uses a higher global id")
        out = out.replace("get_global_id(0)", "gid")
    out = out.replace("get_local_id(0)", "t_tid")
    out = out.replace("get_local_size(0)", "t_tgs")
    out = out.replace("get_group_id(0)", "t_grp")
    out = out.replace("get_num_groups(0)", "t_ngrp")
    out = out.replace(
        "barrier(CLK_LOCAL_MEM_FENCE)", "threadgroup_barrier(mem_flags::mem_threadgroup)"
    )
    out = out.replace(
        "barrier(CLK_GLOBAL_MEM_FENCE)", "threadgroup_barrier(mem_flags::mem_device)"
    )
    out = out.replace("atomic_cmpxchg(", "agx_cmpxchg(")
    out = out.replace("atomic_add(", "agx_atomic_add(")
    out = out.replace("as_float(", "as_type<float>(")
    out = out.replace("as_uint(", "as_type<uint>(")
    out = out.replace("as_int(", "as_type<int>(")
    out = re.sub(
        r"block_sum\(([^(),]+),\s*([^(),]+)\)",
        r"block_sum(\1, \2, t_tid, t_tgs)",
        out,
    )
    out = re.sub(
        r"block_max\(([^(),]+),\s*([^(),]+)\)",
        r"block_max(\1, \2, t_tid, t_tgs)",
        out,
    )
    return out


def emit_kernel(name, params, body):
    is_3d = "get_global_id(1)" in body or "get_global_id(2)" in body
    uses_gid = "get_global_id(" in body
    uses_tid = "get_local_id(" in body or "block_sum(" in body or "block_max(" in body
    uses_tgs = "get_local_size(" in body or "block_sum(" in body or "block_max(" in body
    uses_grp = "get_group_id(" in body
    uses_ngrp = "get_num_groups(" in body
    translated = translate_body(body, is_3d)
    args = []
    local_name = None
    buffer = 0
    for param in split_params(params):
        type_part, pname = split_type_name(param)
        if "__local" in type_part:
            if local_name is not None:
                raise SystemExit(f"{name} has two local buffers")
            if type_part.replace(" ", "") not in ("__localfloat*", "__localconstfloat*"):
                raise SystemExit(f"{name} local is not float*: {type_part}")
            local_name = pname
            continue
        msl = type_part.replace("__global", "device")
        msl = re.sub(r"device\s+const", "const device", msl)
        if "*" in msl:
            args.append(f"{msl} {pname} [[buffer({buffer})]]")
        elif msl in ("int", "long", "float"):
            args.append(f"constant {msl}& {pname} [[buffer({buffer})]]")
        else:
            raise SystemExit(f"{name} unsupported scalar {msl}")
        buffer += 1
    if local_name:
        args.append(f"threadgroup float* {local_name} [[threadgroup(0)]]")
    if uses_gid:
        if is_3d:
            args.append("uint3 gid [[thread_position_in_grid]]")
        else:
            args.append("uint gid [[thread_position_in_grid]]")
    if uses_tid:
        args.append("uint t_tid [[thread_position_in_threadgroup]]")
    if uses_tgs:
        args.append("uint t_tgs [[threads_per_threadgroup]]")
    if uses_grp:
        args.append("uint t_grp [[threadgroup_position_in_grid]]")
    if uses_ngrp:
        args.append("uint t_ngrp [[threadgroups_per_grid]]")
    signature = ",\n    ".join(args)
    return f"kernel void {name}(\n    {signature}) {{{translated}}}\n"


def kernels_from(text):
    found = []
    i = 0
    while True:
        at = text.find("__kernel", i)
        if at < 0:
            break
        paren = text.find("(", at)
        name = text[at + len("__kernel") : paren].replace("void", "").strip()
        close = match_paren(text, paren)
        brace = close + 1
        while text[brace].isspace():
            brace += 1
        if text[brace] != "{":
            raise SystemExit(f"{name} signature is not followed by a body")
        end = match_brace(text, brace)
        found.append((name, text[paren + 1 : close], text[brace + 1 : end]))
        i = end + 1
    return found


def main():
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    text = src.read_text()
    text = strip_ifdef(text, "HAVE_FP64")
    found = kernels_from(text)
    names = [name for name, _, _ in found]
    dropped = {"sum_squares_d", "reduce_sum_d"}
    if dropped & set(names):
        raise SystemExit(f"float64 kernels were not stripped: {dropped & set(names)}")
    pieces = [PREAMBLE]
    for name, params, body in found:
        pieces.append(emit_kernel(name, params, body))
        pieces.append("")
    pieces.append(FILL)
    output = "\n".join(pieces).rstrip() + "\n"
    for token in FORBIDDEN:
        if token in output:
            raise SystemExit(f"leftover token {token}")
    if re.search(r"(?<![\w])atomic_add\(", output):
        raise SystemExit("leftover atomic_add")
    if re.search(r"(?<![\w])barrier\(", output):
        raise SystemExit("leftover barrier")
    OUT.write_text(output)
    print(f"wrote {OUT} kernels={len(found)} fill=3")
    print(" ".join(names))


if __name__ == "__main__":
    main()
