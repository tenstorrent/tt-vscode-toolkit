# ttsim v1.7.3 highlight reel — 6 entries, ~55 second loop
# Embed in ttsim-twenty-and-ten.md as a looping GIF preview

Output assets/img/ttsim-demo.gif

Set Shell "bash"
Set Theme { "name": "Tenstorrent", "black": "#0F2A35", "red": "#FF6B6B", "green": "#27AE60", "yellow": "#F4C471", "blue": "#4FD1C5", "magenta": "#EC96B8", "cyan": "#81E6D9", "white": "#E8F0F2", "brightBlack": "#2D4F5C", "brightRed": "#FF8A80", "brightGreen": "#4EDD8F", "brightYellow": "#F9D789", "brightBlue": "#81E6D9", "brightMagenta": "#F0B8D0", "brightCyan": "#A7EEE7", "brightWhite": "#F5F9FA", "background": "#0F2A35", "foreground": "#E8F0F2", "selectionBackground": "#2D4F5C", "cursor": "#4FD1C5" }
Set FontSize 13
Set Width 960
Set Height 520
Set Padding 18
Set Framerate 24
Set PlaybackSpeed 1.0
Set BorderRadius 6
Set TypingSpeed 35ms

# ── one-time env setup (shown — the user will set these once) ─────────────────
Type "source ~/.tenstorrent-venv/bin/activate"
Enter
Type "export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so"
Enter
Type "export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_SFPLOADMACRO=1"
Enter
Type "export TT_METAL_LOGGER_LEVEL=FATAL"
Enter
Type "cd ~/tt-metal"
Enter
Sleep 600ms

# ── #2: Hello RISC-V ──────────────────────────────────────────────────────────
Type "python3 ~/.figlet_cosmic.py two"
Enter
Sleep 1s
Type "~/.show_kernel.sh tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp"
Enter
Sleep 2.5s
Type "# 2. Hello, RISC-V"
Enter
Sleep 300ms
Type "./build_Release/programming_examples/metal_example_add_2_integers_in_riscv 2>&1 | grep -v '^WARNING:'"
Enter
Sleep 1.2s

# ── #5: compute path ──────────────────────────────────────────────────────────
Type "python3 ~/.figlet_cosmic.py five"
Enter
Sleep 1s
Type "~/.show_kernel.sh tt_metal/programming_examples/add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp"
Enter
Sleep 2.5s
Type "# 5. Same add via TRISC compute engine — kernel compile + dispatch"
Enter
Sleep 300ms
Type "./build_Release/programming_examples/metal_example_add_2_integers_in_compute 2>&1 | grep -v '^WARNING:'"
Enter
Sleep 1.2s

# ── #8: observer effect ───────────────────────────────────────────────────────
Type "python3 ~/.figlet_cosmic.py eight"
Enter
Sleep 1s
Type "~/.show_kernel.sh tt_metal/programming_examples/hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp"
Enter
Sleep 2.5s
Type "# 8. DPRINT changes what gets compiled — without:"
Enter
Sleep 300ms
Type "./build_Release/programming_examples/metal_example_hello_world_datamovement_kernel 2>&1 | grep -v '^WARNING:'"
Enter
Sleep 1.4s
Type "export TT_METAL_DPRINT_CORES=0,0 TT_METAL_DPRINT_RISCVS=BR"
Enter
Sleep 200ms
Type "# with (a different binary is dispatched):"
Enter
Sleep 300ms
Type "./build_Release/programming_examples/metal_example_hello_world_datamovement_kernel 2>&1 | grep -v '^WARNING:'"
Enter
Sleep 1.8s
Type "unset TT_METAL_DPRINT_CORES TT_METAL_DPRINT_RISCVS"
Enter
Sleep 200ms

# ── #15: NoC tile transfer ────────────────────────────────────────────────────
Type "python3 ~/.figlet_cosmic.py fifteen"
Enter
Sleep 1s
Type "~/.show_kernel.sh tt_metal/programming_examples/NoC_tile_transfer/kernels/dataflow/writer0.cpp"
Enter
Sleep 2.5s
Type "# 15. NoC — tile travels from core (0,0) to core (0,1)"
Enter
Sleep 300ms
Type "./build_Release/programming_examples/metal_example_noc_tile_transfer 2>&1 | grep -v '^WARNING:'"
Enter
Sleep 1.2s

# ── #23: multicast ───────────────────────────────────────────────────────────
Type "python3 ~/.figlet_cosmic.py twenty three"
Enter
Sleep 1s
Type "~/.show_kernel.sh tt_metal/programming_examples/contributed/multicast/kernels/dataflow/coordinator_kernel.cpp"
Enter
Sleep 2.5s
Type "# 23. Multicast — one sender, rectangle of receivers"
Enter
Sleep 300ms
Type "export TT_METAL_DPRINT_CORES='(0,0)-(3,0)' TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=0"
Enter
Sleep 200ms
Type "./build_Release/programming_examples/contributed/multicast | grep -E 'PASS|received|multicast'"
Enter
Sleep 3s
Type "unset TT_METAL_DPRINT_CORES TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC"
Enter
Sleep 200ms

# ── #26: profiler ─────────────────────────────────────────────────────────────
Type "python3 ~/.figlet_cosmic.py twenty six"
Enter
Sleep 1s
Type "~/.show_kernel.sh tt_metal/programming_examples/profiler/test_custom_cycle_count_slow_dispatch/test_custom_cycle_count_slow_dispatch.cpp"
Enter
Sleep 2.5s
Type "# 26. Profiler — dispatch overhead vs kernel execution time"
Enter
Sleep 300ms
Type "./build_Release/programming_examples/profiler/test_custom_cycle_count_slow_dispatch"
Enter
Sleep 1.4s

# ── trailer ───────────────────────────────────────────────────────────────────
Type "echo '── 25 more things in the full lesson ──'"
Enter
Sleep 1.5s
Type "python3 ~/.figlet_cosmic.py one more thing..."
Enter
Sleep 3s
