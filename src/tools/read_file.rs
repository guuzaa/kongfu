use crate::tools::ToolHandler;
use kongfu_macros::ToolParams;
use serde::Deserialize;

const MAX_FILE_BYTES: u64 = 100_000; // 100 KB

#[derive(ToolParams, Deserialize)]
pub struct ReadFileParams {
    /// The relative path to the file to read. Must be within the working directory.
    path: String,
}

pub struct ReadFile;

#[async_trait::async_trait]
impl ToolHandler for ReadFile {
    type Params = ReadFileParams;
    type Output = String;

    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the full text contents of a file with line numbers. \
         Only works on UTF-8 text files up to 100 KB. \
         Returns an error for binary files, files that are too large, \
         or paths outside the working directory. \
         \
         **IMPORTANT**: The output includes line numbers to help you identify \
         the exact range to edit. Line numbers are shown as '  1: ' at the start of each line. \
         When using edit_file, use these line numbers to specify your range."
    }

    async fn execute(
        &self,
        params: Self::Params,
    ) -> std::result::Result<Self::Output, Box<dyn std::error::Error>> {
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
        if metadata.len() > MAX_FILE_BYTES {
            return Err(format!(
                "File '{}' is too large ({} bytes, limit is {})",
                params.path,
                metadata.len(),
                MAX_FILE_BYTES
            )
            .into());
        }

        let content = std::fs::read_to_string(&canonical)
            .map_err(|e| format!("Failed to read '{}': {}", params.path, e))?;

        // Add line numbers to help with editing
        let mut result = String::new();
        for (i, line) in content.lines().enumerate() {
            result.push_str(&format!("{:4}: {}\n", i + 1, line));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToFunctionDefinition;

    #[test]
    fn test_read_file_serialization() {
        let tool = ReadFile;
        let def = tool.to_function_definition();
        let def_json = serde_json::to_string_pretty(&def).unwrap();

        // Verify the complete function definition serialization
        assert!(def_json.contains(r#""name": "read_file""#));
        assert!(def_json.contains(r#""description":"#));
        assert!(def_json.contains(r#""parameters":"#));
        assert!(def_json.contains(r#""type": "object""#));
        assert!(def_json.contains(r#""path""#));
        assert!(def_json.contains("relative path"));
        assert!(def_json.contains(r#""required""#));

        // Verify JSON structure
        let parsed: serde_json::Value = serde_json::from_str(&def_json).unwrap();
        assert_eq!(parsed["name"], "read_file");
        assert!(parsed["parameters"]["properties"]["path"]["type"] == "string");
        assert!(
            parsed["parameters"]["required"]
                .as_array()
                .unwrap()
                .contains(&"path".into())
        );
    }

    #[tokio::test]
    async fn test_read_file_with_line_numbers() {
        use tempfile::NamedTempFile;

        // Create a temporary test file in current directory
        let temp_file = NamedTempFile::new_in(".").expect("Failed to create temp file");
        let content = "line 1\nline 2\nline 3";
        std::fs::write(temp_file.path(), content).expect("Failed to write test file");

        let tool = ReadFile;
        let params = ReadFileParams {
            path: temp_file.path().to_str().unwrap().to_string(),
        };

        let result = tool.execute(params).await.unwrap();

        // Verify line numbers are present
        assert!(result.contains("   1: line 1"));
        assert!(result.contains("   2: line 2"));
        assert!(result.contains("   3: line 3"));

        // Verify format: each line should start with "  N: "
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("   1: "));
        assert!(lines[1].starts_with("   2: "));
        assert!(lines[2].starts_with("   3: "));
    }
}
