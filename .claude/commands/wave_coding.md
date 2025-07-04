# /wave_coding

Execute feature development in sequential waves using a single AI agent, with each wave focusing on one main task group.

## Variables

FEATURE_NAME: $ARGUMENTS

## Task File Discovery and Task Group Parsing

**Task File Location:** I will locate the task file at `tasks/tasks-prd-{FEATURE_NAME}.md`.

**Validation Steps:**
1. Verify that `tasks/tasks-prd-{FEATURE_NAME}.md` exists
2. Confirm the file contains a valid task list structure
3. If the file doesn't exist, I will exit with an error message: "Task file not found: tasks/tasks-prd-{FEATURE_NAME}.md. Please generate the task list first using @ai_docs/generate-tasks.mdc"

**Task Group Detection:**
I will parse the task file to identify main task groups (e.g., 1.0, 2.0, 3.0, etc.) and their associated subtasks. Each task group will be executed in a separate wave.

**Task File Contents:** The discovered task file will serve as the implementation plan, executed wave by wave.

## Execute Phase 1: Environment Setup

**Parameter Validation:** Before proceeding, I will validate:
1. `FEATURE_NAME` is provided and not empty
2. Task file `tasks/tasks-prd-{FEATURE_NAME}.md` exists and is readable
3. The `progress/` directory exists (create if it doesn't)

**Safety Check:** I will check for any uncommitted changes in the current directory. If changes are found, I will exit with an error, prompting you to commit or stash them.

**Directory Setup:** I will ensure the `trees/` directory exists at the project root.

**Single Worktree Creation:**
I will create one persistent worktree that will be used across all waves:

1. **Create Worktree:** `git worktree add -b <FEATURE_NAME>-wave ./trees/<FEATURE_NAME>-wave`
2. **Copy Environment:** Copy the `.env` file to `./trees/<FEATURE_NAME>-wave/.env` if it exists.
3. **Setup Task Directory:** 
   - Create `./trees/<FEATURE_NAME>-wave/tasks/` directory
   - Copy the task file from `tasks/tasks-prd-{FEATURE_NAME}.md` to `./trees/<FEATURE_NAME>-wave/tasks/tasks-prd-{FEATURE_NAME}.md`
4. **Setup Progress Tracking:**
   - Ensure `progress/` directory exists at project root
   - Create symlink in worktree: `ln -s ../../progress ./trees/<FEATURE_NAME>-wave/progress`
   - Initialize progress JSON file: `progress/<FEATURE_NAME>-wave.json` with initial status
5. **Install Dependencies:** Change directory to `./trees/<FEATURE_NAME>-wave` and run dependency installation if package files exist.

**Verification:** After setup, I will run `git worktree list` to confirm the worktree was created successfully.

## Execute Phase 2: Wave-by-Wave Development

I will execute development in sequential waves, with each wave focusing on one main task group. After each wave completes, the subagent will report back to the main agent, which will then launch the next wave.

**Wave Orchestration Protocol:**

1. **Parse Task Groups:** Extract all main task groups from the task file (e.g., 1.0, 2.0, 3.0, etc.)
2. **Execute Waves Sequentially:** For each task group:
   - Launch subagent for that specific task group
   - Wait for wave completion
   - Review progress and results
   - Launch next wave for the next task group
3. **Continue Until Complete:** Repeat until all task groups are finished

**Wave Execution Details:**

For each wave, I will:
1. **Identify Current Task Group:** Determine the next incomplete task group (e.g., "2.0 YouTube Transcript Extraction System")
2. **Launch Wave Agent:** Create a focused subagent to work on only that task group
3. **Monitor Progress:** Track completion through progress files
4. **Validate Wave Completion:** Ensure all subtasks in the group are completed
5. **Prepare Next Wave:** Update overall progress and prepare for the next task group

**Working Directory:** 
The subagent will operate in: `trees/<FEATURE_NAME>-wave/`

## Subagent Wave Instructions

**CRITICAL WAVE MISSION:** You are a wave-focused agent working on ONE specific task group within the {FEATURE_NAME} project.

**Your Current Assignment:**
- **Task Group:** {CURRENT_TASK_GROUP} (e.g., "1.0 Project Setup and Dependencies")
- **Subtasks:** {SUBTASK_LIST} (e.g., "1.1, 1.2, 1.3, 1.4, 1.5")
- **Working Directory:** /Users/jeff/Documents/personal/testproject/trees/{FEATURE_NAME}-wave

**MANDATORY WAVE WORKFLOW:**
You MUST complete ALL subtasks within your assigned task group following this exact protocol:

1. **Focus Only on Your Task Group:** Work ONLY on the assigned task group and its subtasks. Do NOT work on other task groups.

2. **Complete All Subtasks in Order:** Work through each subtask in the assigned group:
   - Read the subtask from `tasks/tasks-prd-{FEATURE_NAME}.md`
   - Implement the subtask completely
   - Update `tasks/tasks-prd-{FEATURE_NAME}.md` (mark subtask as `[x]`)
   - Update progress JSON file
   - Create detailed subtask report
   - Git commit changes
   - Move to next subtask in the group

3. **Wave Progress Reporting:**
After completing each subtask, you MUST write to TWO files:

**A. Wave Progress Status File:** `progress/{FEATURE_NAME}-wave.json`
```json
{
  "agent": "{FEATURE_NAME}-wave",
  "current_wave": "1.0",
  "wave_description": "Project Setup and Dependencies",
  "wave_status": "in-progress",
  "current_subtask": "1.3",
  "subtask_description": "Create Dockerfile for containerized deployment",
  "completed_subtasks": ["1.1", "1.2", "1.3"],
  "wave_subtasks": ["1.1", "1.2", "1.3", "1.4", "1.5"],
  "overall_progress": "15%",
  "wave_progress": "60%",
  "last_updated": "2024-01-15T14:30:00Z"
}
```

**B. Subtask Detail Report:** `progress/{FEATURE_NAME}-wave-task-{subtask-number}.md`
```markdown
# Wave {wave-number} - Task {subtask-number} Completion Report

## Wave Context
- **Wave**: {wave-number} - {wave-description}
- **Subtask**: {subtask-number}
- **Description**: {subtask-description}
- **Status**: ✅ Completed
- **Completion Time**: 2024-01-15T14:30:00Z

## Work Performed
- [Detailed description of work completed]
- [Key implementation decisions made]
- [Any architectural considerations]

## Files Modified/Created
- `file1.py` - [Description of changes]
- `file2.py` - [Description of changes]

## Next Steps
- [If more subtasks in wave]: Proceeding to task {next-subtask}: {next-description}
- [If wave complete]: Wave {wave-number} completed. Ready for next wave.

## Issues Encountered
- [Any issues and how they were resolved]

## Testing
- [Testing performed and results]
```

4. **Wave Completion Protocol:**
When ALL subtasks in your assigned task group are complete:
   - Mark the main task group as `[x]` in `tasks/tasks-prd-{FEATURE_NAME}.md`
   - Update wave status to "completed" in progress JSON
   - Create wave summary report: `progress/{FEATURE_NAME}-wave-{wave-number}-summary.md`
   - Make final wave commit: "Complete Wave {wave-number}: {wave-description}"
   - Return control to main agent with completion report

5. **Wave Summary Report:** `progress/{FEATURE_NAME}-wave-{wave-number}-summary.md`
```markdown
# Wave {wave-number} Completion Summary

## Wave Overview
- **Wave**: {wave-number} - {wave-description}
- **Subtasks Completed**: {completed-count} / {total-count}
- **Status**: ✅ Completed
- **Duration**: {start-time} to {end-time}

## Key Accomplishments
- [Major features implemented]
- [Key architectural decisions]
- [Important milestones reached]

## Files Created/Modified
- [Comprehensive list of all files touched]

## Testing Results
- [All tests passing/failing status]
- [Coverage metrics if applicable]

## Issues Resolved
- [Any major issues encountered and resolved]

## Next Wave Preparation
- [Any dependencies or considerations for next wave]
- [Recommended next steps]

## Wave Validation
- [ ] All subtasks marked [x] in task file
- [ ] All subtask reports generated
- [ ] All tests passing
- [ ] Code properly committed
- [ ] Ready for next wave
```

**CRITICAL WAVE RULES:**

1. **Stay in Your Lane:** Only work on your assigned task group. Do NOT touch other task groups.
2. **Complete Before Moving:** Finish ALL subtasks in your wave before reporting completion.
3. **Thorough Reporting:** Every subtask must have detailed progress reporting.
4. **Test Before Completion:** Ensure all implementations are working and tested.
5. **Clean Handoff:** Provide clear completion status for main agent to launch next wave.

**Directory Structure:**
```
trees/{FEATURE_NAME}-wave/
├── tasks/
│   └── tasks-prd-{FEATURE_NAME}.md    # Updated with [x] marks for your wave
├── progress/                          # Shared progress directory (symlink)
├── [implementation files...]
└── [generated by your wave work]
```

## Main Agent Wave Coordination

After each wave completes, I will:

1. **Validate Wave Completion:** Check that all subtasks in the wave are marked complete
2. **Review Progress:** Examine wave summary and progress reports
3. **Update Overall Status:** Update project-wide progress tracking
4. **Prepare Next Wave:** Identify the next task group to work on
5. **Launch Next Wave:** If more waves remain, launch the next wave agent
6. **Final Completion:** When all waves are complete, generate final project summary

**Wave Transition Protocol:**
```
1. Subagent completes wave and reports back
2. Main agent validates wave completion
3. Main agent updates overall progress
4. Main agent identifies next task group
5. Main agent launches next wave subagent
6. Repeat until all task groups complete
```

**Project Completion:**
When all waves are complete, I will:
1. **Final Validation:** Verify all task groups are marked `[x]`
2. **Run Full Test Suite:** Execute comprehensive testing
3. **Generate Final Report:** Create project completion summary
4. **Final Commit:** Make final project commit
5. **Cleanup:** Provide final status and cleanup instructions

**Progress Tracking:**
- **Wave Level:** Each wave tracks its own progress and completion
- **Project Level:** Overall project progress across all waves
- **Detailed Reporting:** Comprehensive documentation of each wave's work

This wave-based approach ensures focused, systematic completion of complex features while maintaining clear progress tracking and quality control at each stage.