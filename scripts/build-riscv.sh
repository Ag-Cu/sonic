#!/bin/bash

# Define the directories
SRC_DIR="native"
TMP_DIR="output/riscv"
OUT_DIR="internal/native/rvv"
TOOL_DIR="tools"

# --- Compiler ---
CC=clang
TARGET_BASE_NAME=""

# --- Set the number of parallel jobs ---
# Use the number of available CPU cores, or default to 4 if nproc is not available.
MAX_JOBS=$(nproc 2>/dev/null || echo 4)
echo "Running with up to $MAX_JOBS parallel jobs."

# 解析参数 (Parameter parsing remains the same)
while [[ $# -gt 0 ]]; do
    case $1 in
        --)
            shift
            if [[ $# -gt 0 ]]; then
                TARGET_BASE_NAME=$1
                shift
            fi
            ;;
        *)
            if [[ "$CC" == "clang" ]] && [[ "$1" != --* ]]; then
                CC=$1
            else
                TARGET_BASE_NAME=$1
            fi
            shift
            ;;
    esac
done

# --- CFLAGS ---
DEFAULT_CFLAGS="-march=rv64gcv_zvl256b -mrvv-vector-bits=zvl -mabi=lp64d"
FINAL_CFLAGS="${CFLAGS:-$DEFAULT_CFLAGS}"

echo "Using Compiler: $CC"
echo "Using CFLAGS: $FINAL_CFLAGS"
if [ -n "$TARGET_BASE_NAME" ]; then
    echo "Targeting specific base name: $TARGET_BASE_NAME"
fi

# Create the output directories if they don't exist
mkdir -p "$TMP_DIR"
mkdir -p "$OUT_DIR"

# --- Main Processing Loop ---
# Counter for active jobs
job_count=0

for src_file in "$SRC_DIR"/*.c; do
    # Check if we need to wait for a job to finish
    if (( job_count >= MAX_JOBS )); then
        wait -n # Waits for the next background job to finish
        job_count=$((job_count - 1))
    fi

    # Group the commands for one file and run them in the background
    (
        base_name=$(basename "$src_file" .c)
        
        # If a specific base_name is given and the current file doesn't match, skip it.
        if [ -n "$TARGET_BASE_NAME" ] && [ "$base_name" != "$TARGET_BASE_NAME" ]; then
            # 'exit' here only exits the subshell, not the main script.
            exit 0
        fi
        
        asm_file="$TMP_DIR/${base_name}.s"

        echo "Compiling $src_file -> $asm_file"

        # Compile C to assembly
        $CC -target riscv64 $FINAL_CFLAGS \
            -Itools/simde/simde \
            -I/workspace/toolchain/sysroot/usr/include \
            -Wno-error -Wno-nullability-completeness \
            -ffreestanding -fno-pic -Os -fno-builtin \
            -fno-exceptions -fno-rtti -fno-stack-protector \
            -fno-asynchronous-unwind-tables -nostdlib \
            -S -o "$asm_file" "$src_file"

        # Check for compilation errors
        if [ $? -ne 0 ]; then
            echo "Error: Compilation failed for $src_file"
            exit 1 # This will exit the subshell with an error status
        fi

        # Convert assembly to Go assembly
        echo "Converting $asm_file to Go assembly for ${base_name}"
        python3 ${TOOL_DIR}/asm2riscv/asm2riscv_split.py ${OUT_DIR}/${base_name}_riscv64.go $asm_file

    ) & # <-- The '&' runs the entire subshell (...) in the background.

    job_count=$((job_count + 1))
done

# Wait for all remaining background jobs to complete before exiting the script
echo "Waiting for all remaining jobs to finish..."
wait
echo "Script finished successfully."