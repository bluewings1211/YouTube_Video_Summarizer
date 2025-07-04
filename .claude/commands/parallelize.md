# /parallelize

Create N git worktrees and run AI agents in each worktree for concurrent development.

## Variables

FEATURE_NAME: $ARGUMENTS
NUMBER_OF_PARALLEL_WORKTREES: $ARGUMENTS

## Task File Discovery and Validation

**Task File Location:** I will locate the task file at `tasks/tasks-prd-{FEATURE_NAME}.md`.

**Validation Steps:**
1. Verify that `tasks/tasks-prd-{FEATURE_NAME}.md` exists
2. Confirm the file contains a valid task list structure
3. If the file doesn't exist, I will exit with an error message: "Task file not found: tasks/tasks-prd-{FEATURE_NAME}.md. Please generate the task list first using @ai_docs/generate-tasks.mdc"

**Task File Contents:** The discovered task file will serve as the implementation plan for all subagents.

## Execute Phase 1: Environment Setup

**Parameter Validation:** Before proceeding, I will validate:
1. `FEATURE_NAME` is provided and not empty
2. `NUMBER_OF_PARALLEL_WORKTREES` is a positive integer
3. Task file `tasks/tasks-prd-{FEATURE_NAME}.md` exists and is readable
4. The `progress/` directory exists (create if it doesn't)

**Safety Check:** I will check for any uncommitted changes in the current directory. If changes are found, I will exit with an error, prompting you to commit or stash them.

**Directory Setup:** I will ensure the `trees/` directory exists at the project root.

> The following steps will be executed in parallel for each worktree to ensure efficiency. I will use absolute paths for all commands.

For each instance from 1 to NUMBER_OF_PARALLEL_WORKTREES, I will:

1.  **Create Worktree:** `git worktree add -b <FEATURE_NAME>-<instance-number> ./trees/<FEATURE_NAME>-<instance-number>`
2.  **Copy Environment:** Copy the `.env` file to `./trees/<FEATURE_NAME>-<instance-number>/.env` if it exists.
3.  **Copy Task File:** Copy the task file from `tasks/tasks-prd-{FEATURE_NAME}.md` to `./trees/<FEATURE_NAME>-<instance-number>/tasks/tasks-prd-{FEATURE_NAME}.md`
4.  **Install Dependencies:** Change directory to `./trees/<FEATURE_NAME>-<instance-number>` and run `uv sync` to install dependencies specified in `uv.lock`.

**Verification:** After setup, I will run `git worktree list` to confirm all worktrees were created successfully.

## Execute Phase 2: Parallel Development

I will now create NUMBER_OF_PARALLEL_WORKTREES independent AI sub-agents. Each agent will work in its own dedicated worktree to build the feature concurrently. This allows for isolated development and testing.

Each sub-agent will operate in its respective directory:
-   Agent 1: `trees/<FEATURE_NAME>-1/`
-   Agent 2: `trees/<FEATURE_NAME>-2/`
-   ...and so on.

**Core Task:**
Each agent will independently and meticulously implement the task list found in `tasks/tasks-prd-{FEATURE_NAME}.md`.

**Implementation Protocol:**
Each sub-agent MUST follow the guidelines in `@ai_docs/process-task-list.mdc`:
- Work on **one sub-task at a time**
- Wait for user permission before starting the next sub-task (unless user says "yolo")
- Update the task list file after completing each sub-task
- Use git for version control: commit after completing each major task
- Mark completed tasks with `[x]` in the task list
- Maintain the "Relevant Files" section accurately

**Progress Tracking:**
Each agent will report its progress by writing to a JSON file in the shared `progress/` directory:

```json
{
  "agent": "<FEATURE_NAME>-<instance-number>",
  "status": "in-progress",
  "step": "parser module implemented",
  "last_updated": "'$(date -Iseconds)'"
}
```
The file will be named `progress/<FEATURE_NAME>-<instance-number>.json`.

**Summary File Generation:**
Upon completion, each agent will generate a summary file `progress/{FEATURE_NAME}-{instance-number}-summary.md` containing:
- Overview of completed tasks
- Key changes made
- Files created/modified
- Test results
- Any issues encountered and resolved
- Usage examples if applicable

**Completion and Validation:**
Upon completing its task, each agent will:

1.  **Generate Report:** Create a comprehensive `RESULTS.md` file in its worktree root, detailing the changes made and providing usage examples if applicable.
2.  **Run Tests:** Execute `.venv/bin/pytest tests/` within its worktree to ensure all tests pass.
3.  **Fix Issues:** Address any test failures, linting errors, or type-checking issues identified.
4.  **Final Commit:** Make a final git commit with all completed changes.
5.  **Generate Summary:** Create a summary file `progress/{FEATURE_NAME}-{instance-number}-summary.md` in the project root progress directory.
6.  **Final Status:** Update its progress file to reflect "completed" status.

This process ensures that we can review multiple, complete, and validated implementations to select the best one for merging into the main branch.
