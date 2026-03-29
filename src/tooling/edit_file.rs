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
        "Replace a range of lines in a file with new content. Can also create new files. \
         Line numbers are 1-indexed and INCLUSIVE (both start and end lines are replaced). \
         \
         **Creating new files**: If the file doesn't exist, use start_line=1 and end_line=1. \
         \
         **Editing existing files**: Always read the file FIRST and identify the EXACT line range. \
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
        // For existing files, canonicalize directly. For new files, canonicalize parent dir.
        let canonical = if std::path::Path::new(&params.path).exists() {
            std::fs::canonicalize(&params.path)
                .map_err(|e| format!("Invalid path '{}': {}", params.path, e))?
        } else {
            // For new files, canonicalize the parent directory and join the filename
            let path = std::path::Path::new(&params.path);
            let parent = path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."));
            let canonical_parent = std::fs::canonicalize(parent)
                .map_err(|e| format!("Invalid path '{}': {}", params.path, e))?;
            let filename = path
                .file_name()
                .ok_or_else(|| format!("Invalid path '{}': no filename", params.path))?;
            canonical_parent.join(filename)
        };

        let cwd = std::fs::canonicalize(".")
            .map_err(|e| format!("Failed to resolve working directory: {}", e))?;
        if !canonical.starts_with(&cwd) {
            return Err(format!(
                "Access denied: '{}' is outside the working directory",
                params.path
            )
            .into());
        }

        // Check if file exists and handle creation
        if !canonical.exists() {
            // File doesn't exist - create it with the replacement content
            if params.start_line != 1 || params.end_line != 1 {
                return Err(format!(
                    "File '{}' does not exist. Can only create new files with start_line=1 and end_line=1. Got start_line={}, end_line={}.",
                    params.path, params.start_line, params.end_line
                )
                .into());
            }

            let new_content = if params.replacement.is_empty() {
                String::new()
            } else {
                params.replacement.clone()
            };

            if new_content.len() > MAX_WRITE_BYTES {
                return Err(format!(
                    "Resulting file would be too large ({} bytes, limit is {})",
                    new_content.len(),
                    MAX_WRITE_BYTES
                )
                .into());
            }

            std::fs::write(&canonical, &new_content)
                .map_err(|e| format!("Failed to create '{}': {}", params.path, e))?;

            let new_line_count = if new_content.is_empty() { 0 } else { new_content.lines().count() };
            return Ok(format!(
                "Successfully created '{}' with {} lines.",
                params.path,
                new_line_count,
            ));
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

        // Special case: empty file - allow writing content at line 1
        if original_line_count == 0 {
            if params.start_line == 1 {
                // Empty file: just write the replacement content
                let new_content = if params.replacement.is_empty() {
                    String::new()
                } else {
                    params.replacement.clone()
                };

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

                let new_line_count = if new_content.is_empty() { 0 } else { new_content.lines().count() };
                return Ok(format!(
                    "Successfully edited '{}': wrote {} lines to empty file.",
                    params.path,
                    new_line_count,
                ));
            } else {
                return Err(format!(
                    "File is empty (0 lines). Cannot start editing from line {}. Only start_line=1 is valid for empty files.",
                    params.start_line
                )
                .into());
            }
        }

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
    use tempfile::NamedTempFile;

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

    #[tokio::test]
    async fn test_edit_file_create_new_file() {
        let new_file_path = NamedTempFile::new_in(".").expect("Failed to create temp file");
        _ = std::fs::remove_file(&new_file_path);

        let tool = EditFile;
        let params = EditFileParams {
            path: new_file_path.path().to_str().unwrap().to_string(),
            start_line: 1,
            end_line: 1,
            replacement: "First line\nSecond line".to_string(),
        };

        let result = tool.execute(params).await.unwrap();

        // Verify success message mentions file creation
        assert!(result.contains("Successfully created"));
        assert!(result.contains("2 lines"));

        // Verify file was created with correct content
        let created_content = std::fs::read_to_string(&new_file_path).unwrap();
        assert_eq!(created_content, "First line\nSecond line");

        // Clean up
        let _ = std::fs::remove_file(&new_file_path);
    }

    #[tokio::test]
    async fn test_edit_file_empty_file() {
        // Create an empty file
        let temp_file = NamedTempFile::new_in(".").expect("Failed to create temp file");
        std::fs::write(temp_file.path(), "").expect("Failed to write empty file");

        let tool = EditFile;
        let params = EditFileParams {
            path: temp_file.path().to_str().unwrap().to_string(),
            start_line: 1,
            end_line: 1,
            replacement: "First line\nSecond line".to_string(),
        };

        let result = tool.execute(params).await.unwrap();

        // Verify success message mentions writing to empty file
        assert!(result.contains("Successfully edited"));
        assert!(result.contains("empty file"));
        assert!(result.contains("2 lines"));

        // Verify file content was written correctly
        let written_content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert_eq!(written_content, "First line\nSecond line");
    }

    #[tokio::test]
    async fn test_edit_file_create_new_file_invalid_range() {
        let new_file_path = NamedTempFile::new_in(".").expect("Failed to create temp file");
        _ = std::fs::remove_file(&new_file_path);

        let tool = EditFile;
        let params = EditFileParams {
            path: new_file_path.path().to_str().unwrap().to_string(),
            start_line: 2,
            end_line: 5,
            replacement: "Content".to_string(),
        };

        let result = tool.execute(params).await;

        // Should fail because creating a new file requires start_line=1 and end_line=1
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("does not exist"));
        assert!(error_msg.contains("start_line=1 and end_line=1"));
    }
}
