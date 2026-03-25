"""
TT-VSCode Bridge - Python helper for VSCode integration

This module provides seamless integration between Python CLI scripts and VSCode,
automatically opening visualization files in the editor.

Usage:
    from tt_vscode_bridge import show_plot

    # ... generate matplotlib plot ...
    plt.savefig('my_plot.png')
    show_plot('my_plot.png')  # Opens in VSCode!

Features:
- Automatically detects if running in VSCode (desktop or code-server)
- Opens images directly in VSCode for viewing
- Falls back gracefully to file-based display if not in VSCode
- Non-blocking - returns immediately after opening
- Works with PNG, JPG, GIF, SVG images

Requirements:
- VSCode (desktop) or code-server (browser-based) - auto-detected
"""

import os
import subprocess
import sys
from typing import Optional


def is_vscode_available() -> bool:
    """
    Check if VSCode is available and accessible.

    Detects both desktop VSCode ('code') and code-server (browser-based).

    Returns:
        True if VSCode command is available, False otherwise
    """
    return _detect_vscode_type() != 'none'


def show_plot(
    image_path: str,
    fallback: bool = True,
    verbose: bool = False
) -> bool:
    """
    Open an image file directly in VSCode.

    Works with both desktop VSCode and code-server (browser-based).

    Args:
        image_path: Absolute or relative path to the image file
        fallback: If True, print file path when VSCode unavailable
        verbose: If True, print status messages

    Returns:
        True if successfully opened in VSCode, False if fallback used

    Example:
        import matplotlib.pyplot as plt

        plt.plot([1, 2, 3], [4, 5, 6])
        plt.savefig('output.png')
        show_plot('output.png')
    """
    # Convert to absolute path
    abs_path = os.path.abspath(image_path)

    # Verify file exists
    if not os.path.exists(abs_path):
        if verbose:
            print(f"⚠️  Image not found: {abs_path}", file=sys.stderr)
        return False

    # Detect VSCode environment
    vscode_type = _detect_vscode_type()

    if vscode_type == 'none':
        if verbose:
            print("ℹ️  VSCode not detected, using fallback display", file=sys.stderr)
        if fallback:
            print(f"📊 View your visualization: {abs_path}")
            print(f"   In VSCode: Cmd/Ctrl+Click the path above")
        return False

    # Handle desktop VSCode (supports --command)
    if vscode_type == 'desktop':
        return _send_to_desktop_vscode(abs_path, fallback, verbose)

    # Handle code-server (browser-based, no --command support)
    if vscode_type == 'code-server':
        return _send_to_code_server(abs_path, fallback, verbose)

    return False


def _detect_vscode_type() -> str:
    """
    Detect which type of VSCode is available.

    Returns:
        'desktop' - Desktop VSCode with --command support
        'code-server' - Browser-based code-server
        'none' - No VSCode detected
    """
    # Try desktop VSCode with --command support
    try:
        result = subprocess.run(
            ['code', '--version'],
            capture_output=True,
            timeout=2
        )
        if result.returncode == 0:
            # Check if it supports --command (desktop) or not (code-server binary)
            help_result = subprocess.run(
                ['code', '--help'],
                capture_output=True,
                timeout=2
            )
            if b'--command' in help_result.stdout or b'--command' in help_result.stderr:
                return 'desktop'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Expected if desktop VSCode is not installed; continue to check for code-server
        pass

    # Try code-server paths
    code_server_paths = [
        '/usr/lib/code-server/lib/vscode/bin/remote-cli/code-server',
        'code-server'
    ]

    for path in code_server_paths:
        try:
            result = subprocess.run(
                [path, '--version'],
                capture_output=True,
                timeout=2
            )
            if result.returncode == 0:
                return 'code-server'
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Expected if this candidate code-server path is unavailable; try the next one.
            pass

    return 'none'


def _send_to_desktop_vscode(abs_path: str, fallback: bool, verbose: bool) -> bool:
    """Send to desktop VSCode by opening the file directly."""
    try:
        # Simply open the file in VSCode
        result = subprocess.run(
            ['code', abs_path],
            capture_output=True,
            timeout=5
        )

        if result.returncode == 0:
            if verbose:
                print(f"✅ Opened in VSCode: {os.path.basename(abs_path)}")
            return True
        else:
            if verbose:
                print("ℹ️  VSCode open failed", file=sys.stderr)
            if fallback:
                print(f"📊 View your visualization: {abs_path}")
            return False

    except Exception as e:
        if verbose:
            print(f"⚠️  Error calling VSCode: {e}", file=sys.stderr)
        if fallback:
            print(f"📊 View your visualization: {abs_path}")
        return False


def _send_to_code_server(abs_path: str, fallback: bool, verbose: bool) -> bool:
    """
    Handle code-server (browser-based VSCode).

    Opens the file directly using code-server command.
    """
    # Try to find code-server binary
    code_server_paths = [
        '/usr/lib/code-server/lib/vscode/bin/remote-cli/code-server',
        'code-server'
    ]

    for code_server_bin in code_server_paths:
        try:
            # Simply open the file in code-server
            result = subprocess.run(
                [code_server_bin, abs_path],
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0:
                if verbose:
                    print(f"✅ Opened in code-server: {os.path.basename(abs_path)}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Try next path
            continue

    # Fallback if all code-server paths failed
    if verbose:
        print("ℹ️  code-server open failed", file=sys.stderr)
    if fallback:
        print(f"📊 View your visualization: {abs_path}")
        print(f"   Cmd/Ctrl+Click the path above to open in VSCode")

    return False


def show_animation_frame(
    frame_path: str,
    frame_number: Optional[int] = None,
    total_frames: Optional[int] = None
) -> bool:
    """
    Display an animation frame in VSCode's Output Preview panel.

    For animations, repeatedly call this with the same frame_path.
    VSCode will auto-refresh as the file is updated.

    Args:
        frame_path: Path to the frame image (reuse same path for animation)
        frame_number: Current frame number (for progress display)
        total_frames: Total number of frames (for progress display)

    Returns:
        True if successfully sent to VSCode, False otherwise

    Example:
        # Game of Life animation
        for step in range(100):
            # ... update game state ...
            plt.imshow(grid)
            plt.savefig('game_frame.png')
            show_animation_frame('game_frame.png', step, 100)
            time.sleep(0.05)  # 20 fps
    """
    success = show_plot(frame_path, fallback=False, verbose=False)

    # Print progress
    if frame_number is not None and total_frames is not None:
        progress = (frame_number / total_frames) * 100
        print(f"🎬 Frame {frame_number}/{total_frames} ({progress:.1f}%)", end='\r')

    return success


def matplotlib_to_vscode(
    save_path: str = 'temp_plot.png',
    dpi: int = 150,
    show_in_vscode: bool = True,
    **savefig_kwargs
) -> str:
    """
    Save current matplotlib figure and optionally show in VSCode.

    This is a convenience function that combines plt.savefig() with show_plot().

    Args:
        save_path: Where to save the figure
        dpi: Resolution for PNG output
        show_in_vscode: If True, send to VSCode preview panel
        **savefig_kwargs: Additional arguments for plt.savefig()

    Returns:
        Absolute path to saved image

    Example:
        import matplotlib.pyplot as plt
        from tt_vscode_bridge import matplotlib_to_vscode

        plt.plot([1, 2, 3], [4, 5, 6])
        plt.title("My Plot")
        matplotlib_to_vscode('my_plot.png')  # Saves AND shows in VSCode!
    """
    import matplotlib.pyplot as plt

    # Save figure
    abs_path = os.path.abspath(save_path)
    plt.savefig(abs_path, dpi=dpi, bbox_inches='tight', **savefig_kwargs)

    # Show in VSCode if requested
    if show_in_vscode:
        show_plot(abs_path, verbose=True)

    return abs_path


# Example usage when run directly
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # Non-GUI backend
    import matplotlib.pyplot as plt

    # Create test plot
    plt.figure(figsize=(8, 6))
    plt.plot([1, 2, 3, 4], [1, 4, 2, 3], 'bo-', linewidth=2, markersize=8)
    plt.title('TT-VSCode Bridge Test', fontsize=14, fontweight='bold')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.grid(True, alpha=0.3)

    # Save and show
    test_path = '/tmp/tt_vscode_bridge_test.png'
    plt.savefig(test_path, dpi=150, bbox_inches='tight')

    print("Testing TT-VSCode Bridge...")
    print(f"Image saved to: {test_path}")

    # Test the bridge
    success = show_plot(test_path, verbose=True)

    if success:
        print("✅ Successfully sent to VSCode Output Preview!")
    else:
        print("ℹ️  Fallback mode - open the file manually")

    print("\nTo use in your scripts:")
    print("  from tt_vscode_bridge import show_plot")
    print("  plt.savefig('my_plot.png')")
    print("  show_plot('my_plot.png')")
