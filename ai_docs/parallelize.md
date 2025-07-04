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
3.  **Setup Task Directory:** 
    - Create `./trees/<FEATURE_NAME>-<instance-number>/tasks/` directory
    - Copy the task file from `tasks/tasks-prd-{FEATURE_NAME}.md` to `./trees/<FEATURE_NAME>-<instance-number>/tasks/tasks-prd-{FEATURE_NAME}.md`
4.  **Setup Progress Tracking:**
    - Ensure `progress/` directory exists at project root
    - Create symlink in worktree: `ln -s ../../progress ./trees/<FEATURE_NAME>-<instance-number>/progress`
    - Initialize progress JSON file: `progress/<FEATURE_NAME>-<instance-number>.json` with initial status
5.  **Install Dependencies:** Change directory to `./trees/<FEATURE_NAME>-<instance-number>` and run dependency installation if package files exist.

**Verification:** After setup, I will run `git worktree list` to confirm all worktrees were created successfully.

## Execute Phase 2: Parallel Development

I will now create NUMBER_OF_PARALLEL_WORKTREES independent AI sub-agents. Each agent will work in its own dedicated worktree to build the feature concurrently. This allows for isolated development and testing.

Each sub-agent will operate in its respective directory:
-   Agent 1: `trees/<FEATURE_NAME>-1/`
-   Agent 2: `trees/<FEATURE_NAME>-2/`
-   ...and so on.

**Core Task:**
Each agent will independently and meticulously implement the task list found in `tasks/tasks-prd-{FEATURE_NAME}.md`.

**CRITICAL INSTRUCTION FOR SUB-AGENTS:**
Each sub-agent MUST immediately begin following the detailed Implementation Protocol below. There is NO EXCEPTION to the progress reporting requirements. Every single sub-task completion MUST be documented with both JSON status updates and detailed markdown reports.

**Implementation Protocol:**
Each sub-agent MUST follow these detailed guidelines:

1. **Task Processing Order**: Work on **one sub-task at a time**, following the exact order in `tasks/tasks-prd-{FEATURE_NAME}.md`

2. **After Each Sub-Task Completion**:
   - **Update Task List**: Mark the completed sub-task as `[x]` in `tasks/tasks-prd-{FEATURE_NAME}.md`
   - **Update Progress**: Write/update the progress JSON file in the shared `progress/` directory
   - **Git Commit**: Commit changes with a descriptive message
   - **Generate Sub-Task Report**: Create a detailed report for this specific sub-task

3. **Sub-Task Progress Reporting**:
Each agent MUST write to TWO files after completing each sub-task:

**A. Progress Status File**: `progress/<FEATURE_NAME>-<instance-number>.json`
```json
{
  "agent": "<FEATURE_NAME>-<instance-number>",
  "status": "in-progress",
  "current_task": "1.2",
  "task_description": "Set up requirements.txt with all necessary dependencies",
  "completed_tasks": ["1.1", "1.2"],
  "total_tasks": 32,
  "last_updated": "2024-01-15T14:30:00Z",
  "estimated_completion": "60%"
}
```

**B. Sub-Task Detail Report**: `progress/<FEATURE_NAME>-<instance-number>-task-{task-number}.md`
```markdown
# Task 1.2 Completion Report

## Sub-Task Details
- **Task ID**: 1.2
- **Description**: Set up requirements.txt with all necessary dependencies
- **Status**: ✅ Completed
- **Completion Time**: 2024-01-15T14:30:00Z

## Work Performed
- Created requirements.txt with all dependencies
- Specified exact versions for compatibility
- Included development dependencies

## Files Modified/Created
- `requirements.txt` - Project dependencies with pinned versions

## Next Steps
- Proceeding to task 1.3: Create Dockerfile for containerized deployment

## Issues Encountered
- None

## Testing
- Verified all dependencies can be installed
- Checked for version conflicts
```

4. **Task List Maintenance**: 
   - Always update `tasks/tasks-prd-{FEATURE_NAME}.md` in the agent's worktree
   - Mark completed sub-tasks with `[x]`
   - When ALL sub-tasks under a parent task are complete, mark the parent task as `[x]`
   - Keep the "Relevant Files" section updated with new files

**Summary File Generation:**
Upon completion, each agent will generate a summary file `progress/{FEATURE_NAME}-{instance-number}-summary.md` containing:
- Overview of completed tasks
- Key changes made
- Files created/modified
- Test results
- Any issues encountered and resolved
- Usage examples if applicable

**Critical Implementation Rules:**

Each sub-agent MUST follow this exact workflow for EVERY sub-task:

```
1. Read the next sub-task from tasks/tasks-prd-{FEATURE_NAME}.md
2. Implement the sub-task completely
3. Update tasks/tasks-prd-{FEATURE_NAME}.md (mark as [x])
4. Update progress/{FEATURE_NAME}-{instance-number}.json
5. Create progress/{FEATURE_NAME}-{instance-number}-task-{task-number}.md
6. Git commit with message: "Complete task {task-number}: {task-description}"
7. Move to next sub-task
```

**IMPORTANT**: The agent MUST NOT skip the progress reporting steps. Each sub-task completion MUST generate both the JSON status update AND the detailed markdown report.

**Directory Structure Requirements:**

Each agent's worktree will have:
```
trees/{FEATURE_NAME}-{instance-number}/
├── tasks/
│   └── tasks-prd-{FEATURE_NAME}.md    # Updated with [x] marks
├── progress/                           # Shared progress directory (symlink to ../../progress/)
├── [implementation files...]
└── RESULTS.md                         # Final summary (generated at end)
```

**Progress Directory Setup:**
Before starting implementation, each agent will:
1. Create a symlink: `ln -s ../../progress ./progress` in their worktree
2. This ensures all progress reports go to the shared progress directory

**Completion and Validation:**
Upon completing ALL tasks, each agent will:

1.  **Final Task List Check:** Verify all tasks in `tasks/tasks-prd-{FEATURE_NAME}.md` are marked `[x]`
2.  **Generate Final Report:** Create comprehensive `RESULTS.md` in worktree root
3.  **Run Tests:** Execute test suite and fix any issues
4.  **Final Progress Update:** Set status to "completed" in progress JSON file
5.  **Generate Summary:** Create `progress/{FEATURE_NAME}-{instance-number}-summary.md` with:
    - Overview of all completed tasks
    - Key changes made
    - Files created/modified  
    - Test results
    - Usage examples
6.  **Final Commit:** Make final git commit with all completed changes

**Validation Checklist:**
- [ ] All sub-tasks marked `[x]` in task file
- [ ] Progress JSON file shows "completed" status  
- [ ] All sub-task detail reports generated
- [ ] RESULTS.md file created
- [ ] Final summary file created
- [ ] All tests passing
- [ ] Final git commit made

This ensures complete traceability and allows review of each agent's step-by-step progress and final implementation.
