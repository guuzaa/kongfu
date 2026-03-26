use crate::tooling::ToolHandler;
use kongfu_macros::ToolParams;
use serde::Deserialize;

const MAX_WRITE_BYTES: usize = 100_000; // 100 KB

#[derive(ToolParams, Deserialize)]
pub struct EditFileParams {
    /// The relative path to the file to edit. Must be within the working directory.
    /// Example: "src/main.rs"
    path: String,
    /// The line number to start replacing (1-indexed, inclusive).
    /// For example, to replace from line 5, set start_line=5.
    start_line: u64,
    /// The line number to stop replacing (1-indexed, inclusive).
    /// For example, to replace through line 10, set end_line=10.
    end_line: u64,
    /// The new content to insert in place of the specified line range.
    /// Can be empty to delete the specified lines.
    /// Example: "fn new_function() {\n    println!(\"Hello\");\n}"
    replacement: String,
}

pub struct EditFile;

#[async_trait::async_trait]
impl ToolHandler for EditFile {
    type Params = EditFileParams;
    type Output = String;

    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replace a range of lines in an existing file with new content. \
         Line numbers are 1-indexed and INCLUSIVE (both start and end lines are replaced). \
         \
         **IMPORTANT**: Always read the file FIRST and identify the EXACT line range. \
         Use read_file first to inspect the current contents and line numbers. \
         Replacement can be empty to delete the specified lines. \
         Only works on UTF-8 text files up to 100 KB."
    }

    async fn execute(
        &self,
        params: Self::Params,
    ) -> std::result::Result<Self::Output, Box<dyn std::error::Error>> {
        // Convert u64 to usize for internal use
        let start_line = params.start_line as usize;
        let end_line = params.end_line as usize;

        // Validate line range before touching the filesystem
        if params.start_line == 0 {
            return Err("start_line must be 1 or greater (lines are 1-indexed)".into());
        }
        if params.start_line > params.end_line {
            return Err(format!(
                "start_line ({}) must be <= end_line ({})",
                params.start_line, params.end_line
            )
            .into());
        }

        // Guard against path traversal
        let canonical = std::fs::canonicalize(&params.path)
            .map_err(|e| format!("Invalid path '{}': {}", params.path, e))?;
        let cwd = std::fs::canonicalize(".")
            .map_err(|e| format!("Failed to resolve working directory: {}", e))?;
        if !canonical.starts_with(&cwd) {
            return Err(format!(
                "Access denied: '{}' is outside the working directory",
                params.path
            )
            .into());
        }

        // Guard against large files
        let metadata = std::fs::metadata(&canonical)
            .map_err(|e| format!("Failed to stat '{}': {}", params.path, e))?;
        if metadata.len() > MAX_WRITE_BYTES as u64 {
            return Err(format!(
                "File '{}' is too large ({} bytes, limit is {})",
                params.path,
                metadata.len(),
                MAX_WRITE_BYTES
            )
            .into());
        }

        // Read the existing file
        let original = std::fs::read_to_string(&canonical)
            .map_err(|e| format!("Failed to read '{}': {}", params.path, e))?;

        let mut lines: Vec<&str> = original.lines().collect();
        let original_line_count = lines.len();

        // Validate line range against actual file length
        if start_line > original_line_count {
            return Err(format!(
                "start_line ({}) exceeds file length ({} lines)",
                params.start_line, original_line_count
            )
            .into());
        }
        if end_line > original_line_count {
            return Err(format!(
                "end_line ({}) exceeds file length ({} lines)",
                params.end_line, original_line_count
            )
            .into());
        }

        // Build replacement lines — empty replacement means deletion
        let replacement_lines: Vec<&str> = if params.replacement.is_empty() {
            vec![]
        } else {
            params.replacement.lines().collect()
        };

        let replacement_line_count = replacement_lines.len();

        // Splice: remove [start_line..=end_line] and insert replacement
        // Convert to 0-indexed for the splice
        let start = start_line - 1;
        let end = end_line; // exclusive upper bound for drain
        lines.drain(start..end);
        for (i, line) in replacement_lines.into_iter().enumerate() {
            lines.insert(start + i, line);
        }

        // Preserve the original trailing newline behaviour
        let mut new_content = lines.join("\n");
        if original.ends_with('\n') {
            new_content.push('\n');
        }

        // Guard against the result being too large
        if new_content.len() > MAX_WRITE_BYTES {
            return Err(format!(
                "Resulting file would be too large ({} bytes, limit is {})",
                new_content.len(),
                MAX_WRITE_BYTES
            )
            .into());
        }

        std::fs::write(&canonical, &new_content)
            .map_err(|e| format!("Failed to write '{}': {}", params.path, e))?;

        Ok(format!(
            "Successfully edited '{}': replaced lines {}-{} ({} lines) with {} lines. \
             File is now {} lines total.",
            params.path,
            params.start_line,
            params.end_line,
            params.end_line - params.start_line + 1,
            replacement_line_count,
            lines.len(),
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::ToFunctionDefinition;

    #[test]
    fn test_edit_file_serialization() {
        let tool = EditFile;
        let def = tool.to_function_definition();
        let def_json = serde_json::to_string_pretty(&def).unwrap();

        // Verify the complete function definition serialization
        assert!(def_json.contains(r#""name": "edit_file""#));
        assert!(def_json.contains(r#""description":"#));
        assert!(def_json.contains(r#""parameters":"#));
        assert!(def_json.contains(r#""type": "object""#));

        // Verify all required parameters
        assert!(def_json.contains(r#""path""#));
        assert!(def_json.contains(r#""start_line""#));
        assert!(def_json.contains(r#""end_line""#));
        assert!(def_json.contains(r#""replacement""#));

        // Verify JSON structure
        let parsed: serde_json::Value = serde_json::from_str(&def_json).unwrap();
        assert_eq!(parsed["name"], "edit_file");

        // Check that all parameters are required
        let required = parsed["parameters"]["required"].as_array().unwrap();
        assert!(required.contains(&"path".into()));
        assert!(required.contains(&"start_line".into()));
        assert!(required.contains(&"end_line".into()));
        assert!(required.contains(&"replacement".into()));

        // Verify parameter types
        assert_eq!(parsed["parameters"]["properties"]["path"]["type"], "string");
        // u64 types should now be serialized as integer
        assert_eq!(
            parsed["parameters"]["properties"]["start_line"]["type"],
            "integer"
        );
        assert_eq!(
            parsed["parameters"]["properties"]["end_line"]["type"],
            "integer"
        );
        assert_eq!(
            parsed["parameters"]["properties"]["replacement"]["type"],
            "string"
        );
    }

    #[tokio::test]
    async fn test_edit_file_functionality() {
        use tempfile::NamedTempFile;

        // Create a temporary test file
        let temp_file = NamedTempFile::new_in(".").expect("Failed to create temp file");
        let initial_content = "line 1\nline 2\nline 3\nline 4\nline 5";
        std::fs::write(temp_file.path(), initial_content).expect("Failed to write test file");

        let tool = EditFile;
        let params = EditFileParams {
            path: temp_file.path().to_str().unwrap().to_string(),
            start_line: 2,
            end_line: 4,
            replacement: "NEW A\nNEW B".to_string(),
        };

        let result = tool.execute(params).await.unwrap();

        // Verify success message
        assert!(result.contains("Successfully edited"));
        assert!(result.contains("replaced lines 2-4 (3 lines) with 2 lines"));

        // Verify file content was correctly edited
        let edited_content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert_eq!(edited_content, "line 1\nNEW A\nNEW B\nline 5");
    }
}
