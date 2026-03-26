use crate::tooling::ToolHandler;
use kongfu_macros::ToolParams;
use serde::Deserialize;

#[derive(ToolParams, Deserialize)]
pub struct ListDirectoryParams {
    /// The directory path to list. Defaults to current directory if not provided.
    /// Must be a relative path within the working directory.
    path: Option<String>,
}

pub struct ListDirectory;

#[async_trait::async_trait]
impl ToolHandler for ListDirectory {
    type Params = ListDirectoryParams;
    type Output = String;

    fn name(&self) -> &str {
        "list_directory"
    }

    fn description(&self) -> &str {
        "List the immediate contents (non-recursive) of a directory. \
         Returns each entry prefixed with [DIR] or [FILE] or [SYMLINK], \
         sorted alphabetically. Defaults to '.' if no path is provided. \
         Use this to explore the file system before reading specific files."
    }

    async fn execute(
        &self,
        params: Self::Params,
    ) -> std::result::Result<Self::Output, Box<dyn std::error::Error>> {
        let path = params.path.unwrap_or_else(|| ".".to_string());

        // Guard against path traversal
        let canonical =
            std::fs::canonicalize(&path).map_err(|e| format!("Invalid path '{}': {}", path, e))?;
        let cwd = std::fs::canonicalize(".")
            .map_err(|e| format!("Failed to resolve working directory: {}", e))?;
        if !canonical.starts_with(&cwd) {
            return Err(
                format!("Access denied: '{}' is outside the working directory", path).into(),
            );
        }

        let mut entries: Vec<String> = std::fs::read_dir(&canonical)
            .map_err(|e| format!("Failed to list directory '{}': {}", path, e))?
            .filter_map(|e| e.ok())
            .map(|entry| {
                let name = entry.file_name().to_string_lossy().to_string();
                let label = match entry.file_type() {
                    Ok(ft) if ft.is_dir() => "DIR",
                    Ok(ft) if ft.is_symlink() => "SYMLINK",
                    _ => "FILE",
                };
                format!("  [{}] {}", label, name)
            })
            .collect();

        entries.sort();

        if entries.is_empty() {
            return Ok("(empty directory)".to_string());
        }

        Ok(entries.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::ToFunctionDefinition;

    #[test]
    fn test_list_directory_serialization() {
        let tool = ListDirectory;
        let def = tool.to_function_definition();
        let def_json = serde_json::to_string_pretty(&def).unwrap();

        // Verify the complete function definition serialization
        assert!(def_json.contains(r#""name": "list_directory""#));
        assert!(def_json.contains(r#""description":"#));
        assert!(def_json.contains(r#""parameters":"#));
        assert!(def_json.contains(r#""type": "object""#));
        assert!(def_json.contains(r#""path""#));
        assert!(def_json.contains("directory path"));

        // Verify JSON structure
        let parsed: serde_json::Value = serde_json::from_str(&def_json).unwrap();
        assert_eq!(parsed["name"], "list_directory");
        assert!(parsed["parameters"]["properties"]["path"]["type"] == "string");
    }
}
