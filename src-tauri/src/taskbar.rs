/// Windows taskbar progress integration
///
/// This module provides functions to set and clear progress on the Windows taskbar. It uses the Windows API to interact with the taskbar and update the progress state based on the provided values.
///
/// # Arguments
/// - `hwnd`: The handle to the window for which the taskbar progress should be updated.
/// - `done`: The amount of work that has been completed.
/// - `total`: The total amount of work to be done. If this is zero, the progress will be set to indeterminate.
///
/// # Example
/// ```rust
/// use crate::taskbar::{set_progress, clear_progress};
///
/// let hwnd = /* obtain window handle */;
/// set_progress(hwnd, 50, 100); // Set progress to 50%
/// clear_progress(hwnd); // Clear progress
/// ```
#[cfg(target_os = "windows")]
pub fn set_progress(hwnd: isize, done: usize, total: usize) {
    use windows::Win32::UI::Shell::{ITaskbarList3, TBPF_NORMAL, TBPF_INDETERMINATE};
    use windows::Win32::Foundation::HWND;
    use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED};
    use windows::core::GUID;

    // CLSID_TaskbarList
    const CLSID_TASKBARLIST: GUID = GUID::from_values(
        0x56FDF344, 0xFD6D, 0x11d0,
        [0x95, 0x8A, 0x00, 0x60, 0x97, 0xC9, 0xA0, 0x90],
    );

    unsafe {
        let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
        let taskbar: Result<ITaskbarList3, _> = CoCreateInstance(
            &CLSID_TASKBARLIST,
            None,
            CLSCTX_INPROC_SERVER,
        );
        if let Ok(taskbar) = taskbar {
            let hwnd = HWND(hwnd as *mut _);
            let _ = taskbar.HrInit();
            if total > 0 {
                let _ = taskbar.SetProgressState(hwnd, TBPF_NORMAL);
                let _ = taskbar.SetProgressValue(hwnd, done as u64, total as u64);
            } else {
                let _ = taskbar.SetProgressState(hwnd, TBPF_INDETERMINATE);
            }
        }
    }
}

/// Clears the progress state on the Windows taskbar for the specified window handle.
///
/// # Arguments
/// - `hwnd`: The handle to the window for which the taskbar progress should be cleared.
///
/// # Example
/// ```rust
/// use crate::taskbar::clear_progress;
///
/// let hwnd = /* obtain window handle */;
/// clear_progress(hwnd); // Clear progress
/// ```
#[cfg(target_os = "windows")]
pub fn clear_progress(hwnd: isize) {
    use windows::Win32::UI::Shell::{ITaskbarList3, TBPF_NOPROGRESS};
    use windows::Win32::Foundation::HWND;
    use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED};
    use windows::core::GUID;

    const CLSID_TASKBARLIST: GUID = GUID::from_values(
        0x56FDF344, 0xFD6D, 0x11d0,
        [0x95, 0x8A, 0x00, 0x60, 0x97, 0xC9, 0xA0, 0x90],
    );

    unsafe {
        let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
        let taskbar: Result<ITaskbarList3, _> = CoCreateInstance(
            &CLSID_TASKBARLIST,
            None,
            CLSCTX_INPROC_SERVER,
        );
        if let Ok(taskbar) = taskbar {
            let hwnd = HWND(hwnd as *mut _);
            let _ = taskbar.HrInit();
            let _ = taskbar.SetProgressState(hwnd, TBPF_NOPROGRESS);
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub fn set_progress(_hwnd: isize, _done: usize, _total: usize) {}

#[cfg(not(target_os = "windows"))]
pub fn clear_progress(_hwnd: isize) {}