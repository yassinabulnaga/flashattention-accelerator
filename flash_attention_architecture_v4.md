# FlashAttention Accelerator — Final Architecture

## Overview

Full matrices Q, K, V (256×64, Q8.8) live in M10K BRAMs on the Cyclone V. Output O (256×64, Q16.16) accumulates in a separate M10K bank. The compute datapath operates on 16×16 tiles loaded into register files. The controller FSM iterates over query row blocks (i=0..15), key/value row blocks (j=0..15), and feature chunks (kk=0..3), loading one 16×16 tile at a time from M10K into registers (16 serial reads), running the existing compute pipeline (GEMM, online softmax, output accumulator) at full speed on register data, then writing results back. No DMA, no external memory, no column banking — just M10K as bulk storage and register files as the parallel-access working set.

## M10K Layout

| BRAM    | Dimensions    | Word width | Depth | Addr bits | M10K blocks |
|---------|--------------|------------|-------|-----------|-------------|
| Q       | 256 × 64     | 16b        | 16384 | 14        | 26          |
| K       | 256 × 64     | 16b        | 16384 | 14        | 26          |
| V_T     | 64 × 256     | 16b        | 16384 | 14        | 26          |
| O       | 256 × 64     | 32b        | 16384 | 14        | 52          |
| **Total** |            |            |       |           | **130/397** |

### Address Mapping (row-major, 16b words)

Q, K: `addr = row * 64 + col` where row ∈ [0,255], col ∈ [0,63].
For tile (i, kk): row = i*16 + r, col = kk*16 + c → `addr = (i*16+r)*64 + kk*16 + c`.

V_T (stored transposed): original V[row,col], stored as V_T[col,row].
`addr = col * 256 + row`. For tile (j, kk): need V[:,j_block] in chunks.
V_T tile (kk, j): row_in_vt = kk*16+c, col_in_vt = j*16+r → `addr = (kk*16+c)*256 + j*16 + r`.

O: `addr = row * 64 + col`, 32b wide. For tile (i, kk): `addr = (i*16+r)*64 + kk*16 + c`.

### Register Files (working tiles)

| Reg file | Size   | Width | Purpose                           |
|----------|--------|-------|-----------------------------------|
| Qi_reg   | 16×16  | 16b   | Current Q tile chunk              |
| Kj_reg   | 16×16  | 16b   | Current K tile chunk              |
| Sij_reg  | 16×16  | 32b   | Score accumulator (across kk)     |
| Vj_reg   | 16×16  | 16b   | Current V_T tile chunk            |
| Oi_reg   | 16×16  | 32b   | Output accumulator chunk          |

Total: 3×(256×16b) + 2×(256×32b) = 12,288 + 16,384 = 28,672 bits ≈ 28.7K FFs.

Plus m[16], ℓ[16] state registers already inside softmax_online.

## Loop Structure

```
for i = 0 to 15:                    // Q row block
  reset m[], ℓ[], Oi accumulators
  for j = 0 to 15:                  // K/V row block
    // Phase A: S[i,j] = Σ_kk Qi[kk] × Kj[kk]^T
    clear Sij_reg
    for kk = 0 to 3:
      load Qi_reg ← M10K Q tile(i, kk)       // 256 reads
      load Kj_reg ← M10K K tile(j, kk)       // 256 reads
      Sij_reg += Qi_reg × Kj_reg^T            // 16×16 dot products

    truncate Sij_reg to Q8.8

    // Phase B: online softmax (per row of Sij)
    for r = 0 to 15:
      softmax_online(Sij_reg[r,:], row_idx=r) → alpha, P̃[r,:]
      rescale Oi across all kk chunks by alpha  // 4 load-rescale-store passes
      write P̃[r,:] back to Sij_reg

    // Phase C: O += P̃ × V
    for kk = 0 to 3:
      load Vj_reg ← M10K V_T tile(kk, j)     // 256 reads
      load Oi_reg ← M10K O tile(i, kk)        // 256 reads
      Oi_reg += P̃_reg × Vj_reg                // 16×16 matmul
      store Oi_reg → M10K O tile(i, kk)        // 256 writes

  // Post-loop: O[i,:] /= ℓ
  for kk = 0 to 3:
    load Oi_reg ← M10K O tile(i, kk)
    for r = 0 to 15:
      recip(ℓ[r]) → scale
      Oi_reg[r,:] *= scale
    store Oi_reg → M10K O tile(i, kk)
```
