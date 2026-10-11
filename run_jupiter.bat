@echo off
setlocal
cd /d "%~dp0"

rem Modes. watch follows the telemetry stream, remote runs on a Modal GPU,
rem mcp exposes the model to an agent over a port, expansion is how a human
rem authorises or judges a proposed growth, and autonomous is the continuous
rem loop from the continuous-operation design. None belongs in the default chain, so they are
rem reached through here rather than by the user finding the script.
if /i "%~1"=="watch"  goto :watch
if /i "%~1"=="remote" goto :remote
if /i "%~1"=="mcp"    goto :mcp
if /i "%~1"=="expansion" goto :expansion
if /i "%~1"=="expand" goto :expansion
if /i "%~1"=="autonomous" goto :autonomous
if /i "%~1"=="auto" goto :autonomous
if /i "%~1"=="organize" goto :organize
if /i "%~1"=="skills" goto :skills
if /i "%~1"=="resources" goto :resources
if /i "%~1"=="status" goto :status
if /i "%~1"=="sweep" goto :sweep
if /i "%~1"=="scrub" goto :scrub
if /i "%~1"=="clean-json" goto :scrub
if /i "%~1"=="purge-digests" goto :purge_digests
if /i "%~1"=="repair-knowledge" goto :purge_digests
if /i "%~1"=="long-tests" goto :long_tests
if /i "%~1"=="tests" goto :long_tests
if /i "%~1"=="compare" goto :compare
if /i "%~1"=="gate-compare" goto :gate_compare
if /i "%~1"=="diagnose" goto :diagnose
if /i "%~1"=="noise" goto :diagnose

echo ========================================
echo          JUPITER MICRO-LEARNING
echo ========================================
echo.

if not exist data\raw mkdir data\raw
if not exist data\curated mkdir data\curated
if not exist out mkdir out

echo [1/9] Verifying the project...
rem The suite is quiet on the console and verbose in a log. Its own output
rem interleaves temporary paths from the fixtures, which is noise in the one
rem window a person is supposed to read the learning report in.
rem
rem run_tests.py rather than `python -m unittest` because the suite is dozens of
rem independent files and the machine has thirty-two cores, so one file per
rem process takes a fraction of the wall clock. The verdict is the same one
rem unittest gives: every file ran, and the chain stops on any failure. Set
rem JUPITER_TESTS_SEQUENTIAL=1 to run them one at a time when a failure needs to
rem be watched rather than read.
rem
rem The long tests are excluded here and are not run by default. They are real
rem cycles of a real model: fifteen files, each building its own corpus, model and
rem store and running three or four cycles, and measured at about a hundred and
rem sixty seconds of the entry point's budget. `run_jupiter.bat long-tests` runs
rem them. Excluding them is a scheduling decision and not a reduction in what is
rem verified: they are run before a change is trusted, and nothing about them is
rem weaker for having been split into one file per test so they can run beside
rem each other.
rem
rem The lines are echoed through unchanged. A window that says nothing for two
rem minutes is indistinguishable from a hung one, which is what the eta exists to
rem stop: each line carries how many files are done and how long the slowest one
rem is expected to take, so the wait visibly counts down.
setlocal enabledelayedexpansion
if "%JUPITER_TESTS_SEQUENTIAL%"=="1" (
  python -m unittest discover -s tests -t . > out\test_report.log 2>&1
) else (
  python run_tests.py --skip-prefix test_longrun > out\test_report.log 2>&1
  type out\test_report.log
)
endlocal
if errorlevel 1 (
  echo.
  echo Tests FAILED. Full output:
  type out\test_report.log
  goto :error
)
for /f "tokens=2 delims= " %%r in ('findstr /c:"Ran" out\test_report.log') do set TEST_COUNT=%%r
echo.
echo       %TEST_COUNT% tests passed across every file. Log: out\test_report.log
set TEST_COUNT=

echo.
echo [2/9] Curating raw text...
python curate_dataset.py
if errorlevel 1 goto :error

echo.
echo [3/9] Inspecting learning material and the tokenizer...
python prepare_dataset.py
if errorlevel 1 goto :error

echo.
echo [4/9] Learning with acceptance and regression detection...
rem The expansion grant is set here rather than left to the shell, because
rem the worker only offers a growth when the safety gate would
rem actually allow it. Without the grant the prompt is suppressed as a
rem question that cannot be answered, and a run that never asks is a run
rem whose expansion control nobody ever sees. Setting it here does not
rem skip the gate: it lets the loop reach the gate and ask.
if "%JUPITER_CONTROL_GRANTS%"=="" set JUPITER_CONTROL_GRANTS=expansion

rem How much of a document one cycle reads. The window count is derived as
rem min(ceiling, target/chunk), so raising only the ceiling does nothing: the
rem target is what actually clamps it. Four times the material per cycle, at
rem four times the time, which is the trade being made - a cycle that reads more
rem learns from more and finishes later.
if "%JUPITER_TARGET_CHARS_PER_CYCLE%"=="" set JUPITER_TARGET_CHARS_PER_CYCLE=384000
if "%JUPITER_MAX_WINDOWS_PER_CYCLE%"=="" set JUPITER_MAX_WINDOWS_PER_CYCLE=32
rem Steps per update, left where it is. More windows means more material per
rem cycle but not more gradient steps, so each window gets proportionally less
rem signal; doubling the steps is the other half of that trade and is a
rem separate decision rather than something to change blind.
if "%JUPITER_MICRO_STEPS%"=="" set JUPITER_MICRO_STEPS=16

python learning_worker.py
if errorlevel 1 goto :error

echo.
echo [5/9] Recent telemetry...
python watch_training.py --summary --limit 8
if errorlevel 1 goto :error

echo.
echo [6/9] Evaluation...
python evaluation.py
if errorlevel 1 goto :error

echo.
echo [7/9] Expansion status and anything waiting on you...
rem Reached from the default chain rather than only from `expansion status`,
rem because the growth approval is a decision a person makes and a decision
rem they cannot see is one they cannot make. The worker asks above when there
rem is a growth to approve; this prints the standing state whether or not it
rem asked, so a run that declined still says what is on record, and a run whose
rem prompt was skipped says why rather than saying nothing.
python expansion.py status
if errorlevel 1 goto :error

echo.
echo [8/9] Remote training availability...
python modal_train.py --check
if errorlevel 1 goto :error

echo.
echo Jupiter learning cycle complete.
echo Read the report above, then choose.
echo.
rem set /p rather than choice, because choice cannot read piped input and an
rem unattended run would stop on "the file is either empty or does not contain
rem the valid choices" instead of finishing. No answer skips the chat, so a
rem non-interactive run reaches the end of the chain rather than hanging or
rem erroring.
set "JUPITER_CHAT="
set /p "JUPITER_CHAT=  Do you want to chat with Jupiter? [Y/N] "
if not defined JUPITER_CHAT goto :done
if /i "%JUPITER_CHAT%"=="Y" goto :chat
if /i "%JUPITER_CHAT%"=="YES" goto :chat
if /i "%JUPITER_CHAT%"=="N" goto :done
if /i "%JUPITER_CHAT%"=="NO" goto :done
echo.
echo "%JUPITER_CHAT%" was not a yes or a no; skipping the conversation.
:done
echo.
echo Cycle finished. Run run_jupiter.bat again to keep learning, or use
echo run_jupiter.bat autonomous to let Jupiter keep learning without you.
exit /b 0

:chat
echo.
echo [9/9] Conversation...
python sample.py
if errorlevel 1 goto :error
exit /b 0

:watch
python watch_training.py
exit /b

:remote
python modal_train.py
exit /b

rem -u is not optional here. Python block-buffers stdout when it is not a
rem terminal, so a server that prints its address and then blocks would appear
rem to print nothing at all when run through this script with output piped or
rem redirected. The operator would see a running process and no way to reach it.
:mcp
python -u mcp_server.py
exit /b

rem JSON punctuation removed from the curated corpus. Curation already scrubs
rem as documents pass through it; this reaches the files that were curated
rem before that existed, because they are already on disk and nothing rewrites
rem them. Nothing is written for a file with nothing to remove.
:scrub
shift
python scrub.py %1 %2 %3 %4 %5 %6
exit /b %errorlevel%

rem Expansion is a human's decision, so it is reached from the entry point
rem rather than by a person discovering they need a Python prompt. Arguments
rem pass straight through, so `run_jupiter.bat expansion approve-growth --loss
rem 4.2` works. The grant is still required: this mode only makes the gate
rem reachable, it does not open it.
:expansion
shift
python expansion.py %1 %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

rem Continuous operation. This is the mode that makes Jupiter an agent rather
rem than a pipeline: it starts once, decides what to do next from its own state,
rem and keeps going until a person stops it with Ctrl+C.
rem
rem The default chain above is still there and is still the right thing to run
rem for a bounded pass, because a chain that finishes says what it did. This
rem mode is for the other case, where the point is that nobody is watching.
rem
rem -u matters here for the same reason it does for mcp: the loop prints its
rem decisions as it makes them, and a buffered print is a decision nobody sees.
rem
rem The grants still apply. A continuous agent that is not granted the domains
rem it needs will keep learning and will keep recording why it did not expand.
if "%JUPITER_TARGET_CHARS_PER_CYCLE%"=="" set JUPITER_TARGET_CHARS_PER_CYCLE=384000
if "%JUPITER_MAX_WINDOWS_PER_CYCLE%"=="" set JUPITER_MAX_WINDOWS_PER_CYCLE=32
if "%JUPITER_CONTROL_GRANTS%"=="" set JUPITER_CONTROL_GRANTS=expansion
:autonomous
echo ========================================
echo   JUPITER CONTINUOUS LEARNING
echo ========================================
echo.
echo This runs until you stop it with Ctrl+C.
echo Set JUPITER_CONTROL_GRANTS to let it act on its own findings, for example:
echo   set JUPITER_CONTROL_GRANTS=expansion
echo which lets it grow the model when a source is short of capacity.
echo Without a grant it keeps learning and records why it did not grow.
echo.
python -u learning_worker.py --continuous
echo.
echo The agent stopped. Its decisions are in out\autonomy.json
pause
exit /b 0

rem What has the agent been doing, and what is it waiting on.
:organize
rem Sorting the corpus into skill folders, and setting duplicate copies aside.
rem Reached from the entry point rather than only as a Python command, because a
rem corpus nobody can navigate is a corpus nobody can reason about picking from.
rem Without --apply it only prints the plan, so the layout can be read before
rem anything moves. Nothing is deleted: every file is moved, and duplicates are
rem kept in data/raw/duplicates rather than removed.
shift
python organize_raw.py %1 %2 %3 %4 %5 %6
exit /b %errorlevel%

:long_tests
rem The long tests, which the default chain does not run.
rem
rem Each of these files is one real multi-cycle run of a real model on a real
rem corpus. They are the only tests that invoke the learning controller end to
rem end, and they are the reason the default run skips them rather than dropping
rem them: a change to the controller is not verified until these pass.
rem
rem Reached from the entry point because the entry point is the only thing a
rem person has to know. `run_jupiter.bat long-tests` runs these; adding `all`
rem runs the whole suite, long ones included.
echo ========================================
echo   JUPITER LONG-RUN TESTS
echo ========================================
echo.
echo Each of these runs a real multi-cycle training run. They take a few minutes.
echo.
python run_tests.py --only-prefix test_longrun
if /i "%~2"=="all" goto :tests_all
exit /b %errorlevel%

:tests_all
echo.
echo Now every test file, long ones included.
echo.
python run_tests.py
exit /b %errorlevel%

:compare
rem Which learns faster: the gradient learner or reward-modulated plasticity.
rem
rem Both start from the same checkpoint, read the same corpus in the same order,
rem run the same number of cycles and are measured on the same quantity, so the
rem only variable is the update rule. It answers which buys more loss reduction
rem per cycle, which is a fact about the learning loop and not a claim about
rem what the model can do.
rem
rem It overwrites out/ with whichever mode ran last, because restoring afterwards
rem would discard what that run learned. Run learning_worker.py again afterwards
rem to carry on from there.
echo ========================================
echo   GRADIENT vs PLASTICITY
echo ========================================
echo.
echo Both modes will run for a few minutes each.
echo out\ will hold whichever mode ran last when it finishes.
echo.
python compare_learning_modes.py
exit /b %errorlevel%

:gate_compare
rem Which acceptance configuration actually judges correctly.
rem
rem Four configurations, each from the same starting weights: no smoothing and
rem one window, smoothing only, more measurement windows only, and both. Each is
rem scored against a fixed evaluation set that is never learned from, so "better"
rem means the model ends in a better place rather than the gate deciding more
rem often.
rem
rem This overwrites out/ with whichever configuration ran last.
echo ========================================
echo   ACCEPTANCE GATE COMPARISON
echo ========================================
echo.
echo Four configurations, several minutes each. Set GATE_CYCLES to change the
echo number of cycles per configuration.
echo.
python compare_gate_configs.py
exit /b %errorlevel%

:diagnose
rem Where the acceptance gate's noise comes from.
rem
rem It answers with the recorded history rather than an argument: how much of
rem the spread is the model and how much is which document happened to be drawn,
rem and whether the spread tracks the size of the gain or is independent of it.
rem Those two findings call for opposite work, so the number is measured rather
rem than guessed and no band is chosen until it has been attributed.
echo ========================================
echo   NOISE DIAGNOSIS
echo ========================================
echo.
python diagnose_noise.py
echo.
echo To measure the spread of the measurement itself, by reading one document
echo through different numbers of windows:
echo   python out\_measure_windows.py
echo.
exit /b 0

:skills
rem What the evaluation says is weak, and how much that moves each folder's
rem documents up the learning queue. Reached from the entry point because the
rem loop this closes is only legible if a person can see both ends of it: which
rem capability is behind, and which material is therefore being preferred.
python skills.py
exit /b %errorlevel%

rem Knowledge repair. A disagreement between two documents used to have a concept
rem invented for each side, and since curation names a curated file by the sha256
rem of its contents, the concept table filled with sixty-eight character hex names
rem carrying no evidence. This removes those and the edges that existed only
rem because of them.
rem
rem Reached from the entry point and not left as a script to run by hand, because a
rem repair nobody can find is a repair nobody runs, and the entry point is the only
rem thing a person has to know.
rem
rem Without --apply it only reports what would go, so the plan can be read before
rem anything is removed. The store is backed up before a write, and a digest that
rem somehow carries evidence stops the run rather than being deleted.
:purge_digests
rem The mode word itself is %1, so the arguments a person typed after it start
rem at %2. Passing %1 through handed argparse the word "purge-digests" as an
rem argument, which it correctly refused.
set "PURGE_ARGS=%2 %3 %4"
echo ========================================
echo   JUPITER KNOWLEDGE REPAIR
echo ========================================
echo.
echo This reports what it would remove. Nothing is written without --apply.
echo.
python purge_digest_concepts.py %PURGE_ARGS%
if "%~2"=="apply" goto :purge_apply
echo.
echo Re-run with apply to remove them, for example:
echo   run_jupiter.bat purge-digests apply
exit /b 0

:purge_apply
echo.
python purge_digest_concepts.py --apply
exit /b %errorlevel%

:resources
rem What the machine is doing and where the work would run. Reached from the
rem entry point because the decision is made once at startup and then holds for
rem every hour of a long run, so a person who wants to know why a run is slow
rem should not have to guess at it.
python resources.py
exit /b %errorlevel%

:status
python expansion.py status
echo.
python autonomy.py
echo.
python evaluation.py
exit /b 0

rem The micro_steps sweep. Each setting runs from the same checkpoint and
rem the same corpus so the only variable is the step count. It overwrites
rem out\Jupiter_latest.safetensors as it goes, and restores the model it
rem started from when it finishes.
:sweep
python -u sweep_micro_steps.py
exit /b %errorlevel%

:error
echo.
echo Jupiter stopped because an error occurred.
pause
exit /b 1
