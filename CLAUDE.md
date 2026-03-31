# CLAUDE.md

## User stories

When the user mentions **user stories**, **US-##**, or **sprint scope**, treat **[GitHub Project #3](https://github.com/users/yongggquannn/projects/3)** as the canonical backlog. Open it (or the linked issues) before inferring requirements; do not invent story details from memory.

## Documentation

Whenever you **add a new script, DAG, or meaningful file**, update **README.md** in the same change: a short “how to run / how to verify” subsection so others can install deps, run commands, and confirm it works.

## Verification

After implementing a change, **run tests or the documented verification steps** (from README or the project’s usual commands) to confirm the change works end-to-end. Do not treat the task as complete until that check passes or any failure is explained and addressed.
