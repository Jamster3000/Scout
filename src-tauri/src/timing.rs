/// Debug timing utilities.
///
/// All macros compile to zero-cost no ops in release builds.
///
/// # Macros
/// - `time_block!(label, expr)` - times a single expression
/// - `time_start!(var)` - records a start instant into `var`
/// - `time_end!(var, label)` - prints elapsed ms since `var` and resets it
///
/// # Example
/// ```rust
/// use crate::time_block;
/// let result = time_block!("my operation", do_work());
///
/// use crate::{time_start, time_end};
/// time_start!(t);
/// step_one();
/// time_end!(t, "step one");
/// step_two();
/// time_end!(t, "step two");
/// ```

#[cfg(debug_assertions)]
#[macro_export]
macro_rules! time_block {
	($name:expr, $block:expr) => {{
		let _t_start = std::time::Instant::now();
		let _t_result = $block;
		let _elapsed = _t_start.elapsed();
		let _ms = _elapsed.as_secs_f64() * 1000.0;
		
		if _ms < 0.01 {
			eprintln!("[timing] {}: {:?}", $name, _elapsed);
		} else if _ms >= 1000.0 {
			eprintln!("[timing] {}: {:.2}s", $name, _elapsed.as_secs_f64());
		} else {
			eprintln!("[timing] {}: {:.2}ms", $name, _ms);
		}
		_t_result
	}};
}

#[cfg(debug_assertions)]
#[macro_export]
macro_rules! time_start {
	($var:ident) => {
		let $var = std::time::Instant::now();
	};
}

#[cfg(debug_assertions)]
#[macro_export]
macro_rules! time_end {
	($var:ident, $name:expr) => {
		let _elapsed = $var.elapsed();
		let _ms = _elapsed.as_secs_f64() * 1000.0;
		
		if _ms < 0.01 {
			eprintln!("[timing] {}: {:?}", $name, _elapsed);
		} else if _ms >= 1000.0 {
			eprintln!("[timing] {}: {:.2}s", $name, _elapsed.as_secs_f64());
		} else {
			eprintln!("[timing] {}: {:.2}ms", $name, _ms);
		}
		#[allow(unused_variables)]
		let $var = std::time::Instant::now();
	};
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! time_block {
    ($name:expr, $block:expr) => {
        $block
    };
}
 
#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! time_start {
    ($var:ident) => {};
}
 
#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! time_end {
    ($var:ident, $name:expr) => {};
}