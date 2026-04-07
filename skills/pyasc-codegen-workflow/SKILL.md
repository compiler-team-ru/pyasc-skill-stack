---
name: pyasc-codegen-workflow
description: pyasc kernel development standard workflow. Contains 4 phases with checkpoints — environment preparation, design, implementation with review, and verification. Trigger — user requests kernel development using pyasc. Applicable to pyasc JIT kernels (Python scripts), not Ascend C direct mode.
---

# pyasc kernel development workflow

## pyasc JIT kernel requirements

> **This skill is for pyasc JIT kernels**: the final product is a runnable Python script using `@asc.jit`, not a C++ executable or `.so` library.

**Phase 2 exit requirements**:
- There is a kernel function decorated with `@asc.jit`
- There is a launch function using `kernel[core_num, stream](...)`
- There is a host-side driver (`if __name__ == "__main__"`) or test harness
- There is output verification (`torch.allclose` or numpy comparison)
- The kernel runs successfully (on NPU or Model backend)

**Golden reference**: `golden/tutorials/01_add.py` (local) or `~/workspace/pyasc/python/tutorials/01_add/add.py` (external)

---

> **Forced Workflow**: Phase 0 -> Phase 1 -> Phase 2 -> Phase 3
>
> **Forbidden**: Write code directly, skip design, skip acceptance review

---

## Process Integrity Checklist

| Phase | Required Task | Count | Why |
|-------|---------------|-------|-----|
| Phase 0 | Initialize project + verify environment | 1 | Ensure environment is ready and directory structure is complete |
| Phase 1 | Design + evaluation | 2 | Evaluation finds design flaws early |
| Phase 2 | Implementation + review + acceptance | 2/branch | Acceptance catches syntax violations and API misuse |
| Phase 3 | Verification | On demand | Ensure kernel produces correct output |

**Common mistakes**:
- Phase 0 skips environment check -> pyasc not installed, CANN missing
- Phase 2 uses unsupported syntax inside `@asc.jit` -> JIT compilation fails
- Skipping acceptance review -> syntax constraint violations missed
- Using fixed `core_num=8` / `tile_num=8` with small shapes (e.g. `[1,128]`) -> tile_length too small -> silent incorrect results on simulator. **Always use `_compute_tiling()` and pass `tile_num` as `asc.ConstExpr[int]`** (pyasc's JIT cache ignores ordinary globals — only `ConstExpr` values are cache-safe).

---

## Quick Checklist

- [ ] Phase 0: Project directory + environment.json saved? (CP-0)
- [ ] Phase 1: Design document + rating >= 8.5? (CP-1)
- [ ] Phase 2: Acceptance review report + verification passed? (CP-2)
- [ ] Phase 3: Test pass record? (CP-3)

---

## Process Overview

```
Phase 0: Environment preparation         [1 Task]
    |
Phase 1: Design and API selection        [2 Tasks]
    |
Phase 2: Kernel implementation           [2 Tasks/branch]
    |
Phase 3: Verification and delivery
```

---

## Force checkpoints

| Checkpoint | Timing | Check content | Passing criteria |
|------------|--------|---------------|------------------|
| **CP-0** | After Phase 0 | Project directory + environment.json | Directory exists + file complete |
| CP-1 | After Phase 1 | design.md + rating | 2 Task records + rating >= 8.5 |
| CP-2 | After Phase 2 per branch | Acceptance report + verification | 2 Task records + score >= 8.5 |
| CP-3 | After Phase 3 | Verification record | Tests passed |

---

## Phase 0: Environment preparation

> **PROHIBITED**: Skip Phase 0 and go directly to design

### Step 1: Initialize kernel project

```bash
bash <skills_path>/pyasc-codegen-workflow/scripts/init_kernel_project.sh {kernel_name}
```

**Created directory structure**:
```
kernels/{kernel_name}/
├── docs/           # Design documents
├── test/           # Test data and scripts
├── kernel.py       # Kernel implementation (created in Phase 2)
└── README.md       # Kernel description
```

### Step 2: Verify environment and save results

```bash
bash <skills_path>/pyasc-codegen-workflow/scripts/verify_environment.sh {kernel_name}
```

**Output**: `kernels/{kernel_name}/docs/environment.json`

### CP-0 Exit Conditions

- [ ] Project directory created (`kernels/{kernel_name}/`)
- [ ] Subdirectories created (docs/, test/)
- [ ] **environment.json saved** (including Python version, pyasc version, CANN version, backend)

**Detailed guide**: [references/phase0-environment.md](references/phase0-environment.md)

---

## Phase 1: Design and API selection

> **Prerequisites**: Phase 0 completed, project directory and environment.json exist

```
Main Agent
 |-- Task 1: Design -> Steps 1-6 -> Design document
 |-- Task 2: Evaluation -> Score >= 8.5 -> Write score into design.md footer
```

### Design steps (ALL MANDATORY)

1. **Understand the operation**: What mathematical/logical operation does this kernel perform?
2. **Retrieve documentation**: Read the golden API reference for the target operation.
   - **MANDATORY READ**: `golden/docs/python-api/language/generated/asc.language.basic.{operation}.md`
   - Also read the closest tutorial: `golden/tutorials/01_add.py` or another relevant golden tutorial
3. **Select APIs**: Choose from `asc.language.basic`, `asc.language.core`, etc.
   - **MANDATORY READ**: Load skill `pyasc-api-patterns` to check correct API usage patterns
   - **CRITICAL**: Read the "Dynamic tiling for variable shapes" section — you MUST use `_compute_tiling()` if any requested shape has fewer than ~4096 elements (e.g. `[1,128]`)
4. **Check syntax constraints**: Verify all constructs you plan to use are supported.
   - **MANDATORY READ**: Load skill `pyasc-syntax-constraints` — confirm every construct is in the supported set
5. **Write design document**: Use [templates/design-template.md](templates/design-template.md)
   - Check all boxes in "Syntax compliance check" section — every box must be checked `[x]`
6. **Evaluate design**: Rate the design on a 10-point scale. Write a "## Design Score" section at the bottom of design.md with a numeric rating.
   - Rating must be >= 8.5 to pass CP-1

### CP-1 Exit Conditions

- [ ] design.md exists in `kernels/{name}/docs/`
- [ ] "## Design Score" section present with numeric rating >= 8.5
- [ ] All syntax compliance checkboxes are `[x]`
- [ ] Evidence of reading `pyasc-syntax-constraints` and `pyasc-api-patterns` skills
- [ ] Evidence of reading the golden API reference for the target operation

**Detailed guide**: [references/phase1-design.md](references/phase1-design.md)

---

## Phase 2: Kernel implementation

### Execution process — TWO SEPARATE TASKS REQUIRED

> **PROHIBITED**: Writing kernel.py and acceptance_review.md in the same uninterrupted sequence.
> Task 1 must be complete (kernel written + self-review) BEFORE starting Task 2 (acceptance).

```
Task 1: Implementation + self-review
    -> Read kernel template: templates/kernel-template.py
    -> Implement kernel.py
    -> Self-check against pyasc-syntax-constraints (re-read the skill)
    -> Self-check against pyasc-api-patterns (re-read the skill)
    -> Write self_review.md with pass/fail for each check item
    -> Return report

Task 2: Acceptance review (MUST be a separate step after Task 1)
    -> MANDATORY: Load skill pyasc-code-review
    -> Review kernel.py using the pyasc-code-review checklist
    -> Rate on 10-point scale
    -> Write acceptance_review.md with "## Total Score" section
    -> Score >= 8.5? -> Proceed to Phase 3
    -> Score < 8.5? -> Fix issues, re-do Task 2
```

### Phase 2 mandatory tool calls

| Step | What to do | Artifact |
|------|-----------|----------|
| Read template | Read `templates/kernel-template.py` | — |
| Implement | Write `kernel.py` | `kernels/{name}/kernel.py` |
| Self-review | Re-read `pyasc-syntax-constraints` + `pyasc-api-patterns`, check kernel against them | `kernels/{name}/docs/self_review.md` |
| Acceptance | Load `pyasc-code-review` skill, apply its checklist to kernel.py | `kernels/{name}/docs/acceptance_review.md` |

### CP-2 Exit Conditions (all must be met)

| Condition | Check method | When not met |
|-----------|-------------|--------------|
| kernel.py written | File exists | Implement kernel |
| self_review.md written | File exists | Write self-review |
| acceptance_review.md written | File exists with "## Total Score" | Run acceptance |
| Score >= 8.5 | Check score value | Fix and re-accept |
| `pyasc-code-review` skill was loaded/read | Evidence in session | Load the skill |

**Detailed guide**: [references/phase2-implementation.md](references/phase2-implementation.md)

---

## Phase 3: Verification and delivery

> **PROHIBITED**: Skipping Phase 3. Even if runtime is unavailable, you MUST produce a verification record.

### Phase 3 steps (ALL MANDATORY)

> **TIME BUDGET**: Phase 3 should take at most 2-3 tool calls. If runtime fails on the first attempt,
> do NOT debug the runtime environment. Record the error and proceed to static verification immediately.

Verification has three layers:

1. **Layer 1 — Simulator execution** (use `python3.10` — the python with pyasc and torch installed):
   ```bash
   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
   cd kernels/{name}
   python3.10 kernel.py -r Model -v Ascend910B1
   ```
   - **IMPORTANT**: Do NOT use bare `python` or `python3` — those may resolve to a different version without pyasc/torch.
   - **IMPORTANT**: The `LD_LIBRARY_PATH` export and `-v Ascend910B1` platform flag are required for the CANN simulator.
   - If this succeeds, record the output in `kernels/{name}/docs/verification.md`
   - If runtime fails for ANY reason (missing lib, platform error, timeout, etc.): record the error message and **immediately proceed to Layer 2**. Do NOT attempt to debug or fix the runtime environment.

2. **Layer 2 — Static AST verification** (always do this even if runtime works):
   - Parse kernel.py with Python `ast` module to verify it is valid Python
   - Verify `@asc.jit` decorator is present
   - Verify no banned constructs (`print`, `try/except`, `break`, `continue`, `lambda`, `import` inside JIT)
   - Verify `set_flag`/`wait_flag` sync pairs present
   - Verify `data_copy` usage present
   - Verify `allclose` or numpy verification present in host code

3. **Layer 3 — Write verification record** — `kernels/{name}/docs/verification.md`:
   - Runtime result (PASS / FAIL / SKIP with reason and error message)
   - Static verification results (each check: PASS/FAIL)
   - Limitation statement if runtime was skipped or failed

### CP-3 Exit Conditions

| Condition | Check method | When not met |
|-----------|-------------|--------------|
| Kernel execution attempted | Evidence of running `python3.10 kernel.py` OR documented skip reason | Run or document |
| verification.md written | File exists | Write verification record |
| Static checks passed | All checks listed in verification.md | Fix kernel |

**Detailed guide**: [references/phase3-testing.md](references/phase3-testing.md)

---

## Quick index

### Templates
- Design document: [templates/design-template.md](templates/design-template.md)
- Kernel template: [templates/kernel-template.py](templates/kernel-template.py)

### Scripts
- `init_kernel_project.sh` — initialize kernel directory
- `verify_environment.sh` — environment verification

### Mandatory skills (must be loaded during workflow)

| Skill | When to load | Phase |
|-------|-------------|-------|
| `pyasc-syntax-constraints` | Before writing any kernel code | Phase 1 step 4, Phase 2 self-review |
| `pyasc-api-patterns` | Before selecting APIs | Phase 1 step 3, Phase 2 self-review |
| `pyasc-code-review` | During acceptance review | Phase 2 Task 2 |

### Recommended skills
- `pyasc-docs-search` — Documentation search (Phase 1 step 2)
- `pyasc-build-run-verify` — Build and verify (Phase 3)
- `pyasc-env-check` — Environment check (Phase 0)
- `pyasc-task-focus` — Task tracking for long workflows
