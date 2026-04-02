use crate::error::{KongfuError, Result};

pub struct HttpClient {
    http: reqwest::Client,
    api_key: Option<String>,
    pub base_url: String,
}

impl HttpClient {
    pub fn new(api_key: Option<String>, base_url: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key,
            base_url,
        }
    }

    pub fn endpoint(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }

    pub async fn post(&self, path: &str, body: &serde_json::Value) -> Result<reqwest::Response> {
        let req = if let Some(api_key) = &self.api_key {
            self.http
                .post(self.endpoint(path))
                .header("Authorization", format!("Bearer {}", api_key))
        } else {
            self.http.post(self.endpoint(path))
        };

        let response = req
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| KongfuError::NetworkError(format!("API request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(KongfuError::ApiError {
                status,
                message: error_text,
            });
        }

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_new_with_api_key() {
        let client = HttpClient::new(
            Some("test-key".to_string()),
            "https://api.example.com".to_string(),
        );
        assert_eq!(client.api_key, Some("test-key".to_string()));
        assert_eq!(client.base_url, "https://api.example.com");
    }

    #[test]
    fn test_new_without_api_key() {
        let client = HttpClient::new(None, "https://api.example.com".to_string());
        assert_eq!(client.api_key, None);
        assert_eq!(client.base_url, "https://api.example.com");
    }

    #[test]
    fn test_endpoint_trailing_slash() {
        let client = HttpClient::new(None, "https://api.example.com/".to_string());
        assert_eq!(
            client.endpoint("chat/completions"),
            "https://api.example.com/chat/completions"
        );
    }

    #[test]
    fn test_endpoint_without_trailing_slash() {
        let client = HttpClient::new(None, "https://api.example.com".to_string());
        assert_eq!(
            client.endpoint("chat/completions"),
            "https://api.example.com/chat/completions"
        );
    }

    #[test]
    fn test_endpoint_with_leading_slash() {
        let client = HttpClient::new(None, "https://api.example.com".to_string());
        assert_eq!(
            client.endpoint("/chat/completions"),
            "https://api.example.com/chat/completions"
        );
    }

    #[test]
    fn test_endpoint_both_slashes() {
        let client = HttpClient::new(None, "https://api.example.com/".to_string());
        assert_eq!(
            client.endpoint("/chat/completions"),
            "https://api.example.com/chat/completions"
        );
    }

    #[test]
    fn test_base_url_public() {
        let client = HttpClient::new(None, "https://api.example.com".to_string());
        assert_eq!(client.base_url, "https://api.example.com");
    }

    #[tokio::test]
    async fn test_post_success_with_api_key() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .match_header("content-type", "application/json")
            .with_status(200)
            .with_body(r#"{"result": "success"}"#)
            .create_async()
            .await;

        let client = HttpClient::new(Some("test-key".to_string()), server.url());
        let body = json!({"model": "gpt-4", "messages": []});

        let response = client.post("chat/completions", &body).await.unwrap();
        assert_eq!(response.status(), 200);

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_post_success_without_api_key() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/api/chat")
            .match_header("content-type", "application/json")
            .with_status(200)
            .with_body(r#"{"result": "success"}"#)
            .create_async()
            .await;

        let client = HttpClient::new(None, server.url());
        let body = json!({"model": "llama3", "messages": []});

        let response = client.post("api/chat", &body).await.unwrap();
        assert_eq!(response.status(), 200);

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_post_error_response() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body(r#"{"error": "unauthorized"}"#)
            .create_async()
            .await;

        let client = HttpClient::new(Some("wrong-key".to_string()), server.url());
        let body = json!({"model": "gpt-4", "messages": []});

        let result = client.post("chat/completions", &body).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            KongfuError::ApiError { status, message } => {
                assert_eq!(status, 401);
                assert!(message.contains("unauthorized"));
            }
            _ => panic!("Expected ApiError"),
        }

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_post_with_request_body() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/completions")
            .match_body(r#"{"model":"test-model","prompt":"hello"}"#)
            .with_status(200)
            .with_body(r#"{"choices":[]}"#)
            .create_async()
            .await;

        let client = HttpClient::new(Some("key".to_string()), server.url());
        let body = json!({"model": "test-model", "prompt": "hello"});

        let response = client.post("completions", &body).await.unwrap();
        assert_eq!(response.status(), 200);

        let json: serde_json::Value = response.json().await.unwrap();
        assert_eq!(json["choices"].as_array(), Some(&vec![]));

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_post_different_status_codes() {
        let mut server = mockito::Server::new_async().await;

        // Test 400 Bad Request
        let mock_400 = server
            .mock("POST", "/error/400")
            .with_status(400)
            .with_body(r#"{"error": "bad request"}"#)
            .create_async()
            .await;

        let client = HttpClient::new(Some("key".to_string()), server.url());
        let body = json!({});

        let result = client.post("error/400", &body).await;
        assert!(result.is_err());

        if let KongfuError::ApiError { status, .. } = result.unwrap_err() {
            assert_eq!(status, 400);
        } else {
            panic!("Expected ApiError with status 400");
        }

        mock_400.assert_async().await;

        // Test 500 Internal Server Error
        let mock_500 = server
            .mock("POST", "/error/500")
            .with_status(500)
            .with_body(r#"{"error": "internal server error"}"#)
            .create_async()
            .await;

        let result = client.post("error/500", &body).await;
        assert!(result.is_err());

        if let KongfuError::ApiError { status, .. } = result.unwrap_err() {
            assert_eq!(status, 500);
        } else {
            panic!("Expected ApiError with status 500");
        }

        mock_500.assert_async().await;
    }
}
