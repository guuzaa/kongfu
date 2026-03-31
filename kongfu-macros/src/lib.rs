use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Attribute, Data, DataStruct, DeriveInput, Field, Fields, FnArg, ItemFn, Pat, PatType, Type,
    parse_macro_input,
};

/// Derive macro for tool parameters
/// Generates JSON schema and validation logic
#[proc_macro_derive(ToolParams, attributes(param))]
pub fn derive_tool_params(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let _name = &input.ident;

    let schema_impl = generate_schema_impl(&input);

    let expanded = quote! {
        // Implement Schema trait for this struct
        #schema_impl
    };

    TokenStream::from(expanded)
}

/// Get the path to the ToolParams trait for derive macro (internal use)
fn get_tool_params_path_derive() -> TokenStream2 {
    // Use crate::ToolParams - works within the kongfu library
    quote! { crate::ToolParams }
}

/// Generate JSON schema implementation
fn generate_schema_impl(input: &DeriveInput) -> TokenStream2 {
    let name = &input.ident;
    let tool_params_path = get_tool_params_path_derive();

    let fields = match &input.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => &fields.named,
        _ => {
            return quote! {
                compile_error!("ToolParams only supports structs with named fields");
            };
        }
    };

    let properties: Vec<TokenStream2> = fields.iter().map(generate_property).collect();

    let required: Vec<TokenStream2> = fields
        .iter()
        .filter(|field| !is_optional(field))
        .map(|field| {
            let name = field.ident.as_ref().unwrap();
            let name_str = name.to_string();
            quote! { #name_str }
        })
        .collect();

    quote! {
        impl #tool_params_path for #name {
            fn schema() -> serde_json::Value {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        #( #properties ),*
                    },
                    "required": [ #( #required ),* ]
                })
            }

            fn from_value(value: serde_json::Value) -> std::result::Result<Self, String>
            where
                Self: Sized + for<'de> serde::Deserialize<'de>,
            {
                serde_json::from_value(value)
                    .map_err(|e| format!("Failed to parse parameters: {}", e))
            }
        }
    }
}

/// Generate schema property for a field
fn generate_property(field: &Field) -> TokenStream2 {
    let name = field.ident.as_ref().unwrap();
    let name_str = name.to_string();

    // Get description from #[param] or doc comment
    let description = extract_description(field);

    let (type_name, _is_optional) = get_field_type_info(field);

    match type_name.as_str() {
        "String" => {
            quote! {
                #name_str: {
                    "type": "string",
                    "description": #description
                }
            }
        }
        "i32" | "i64" | "u32" | "u64" => {
            quote! {
                #name_str: {
                    "type": "integer",
                    "description": #description
                }
            }
        }
        "f32" | "f64" => {
            quote! {
                #name_str: {
                    "type": "number",
                    "description": #description
                }
            }
        }
        "bool" => {
            quote! {
                #name_str: {
                    "type": "boolean",
                    "description": #description
                }
            }
        }
        _ => {
            // Default to string for unknown types
            quote! {
                #name_str: {
                    "type": "string",
                    "description": #description
                }
            }
        }
    }
}

/// Extract description from field attributes
fn extract_description(field: &Field) -> String {
    // First try doc comments
    for attr in &field.attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(nv) = &attr.meta {
                if let syn::Expr::Lit(expr) = &nv.value {
                    if let syn::Lit::Str(s) = &expr.lit {
                        return s.value().trim().to_string();
                    }
                }
            }
        }
    }

    // Use a default description
    let field_name = field.ident.as_ref().unwrap().to_string();
    format!("The {} parameter", field_name)
}

/// Get field type information
fn get_field_type_info(field: &Field) -> (String, bool) {
    let ty = &field.ty;

    // Check if it's Option<T>
    if let Type::Path(type_path) = ty {
        if let Some(seg) = type_path.path.segments.last() {
            if seg.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    if let Some(arg) = args.args.first() {
                        if let syn::GenericArgument::Type(inner_type) = arg {
                            if let Type::Path(inner_path) = inner_type {
                                if let Some(inner_seg) = inner_path.path.segments.last() {
                                    return (inner_seg.ident.to_string(), true);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Non-optional type
    if let Type::Path(type_path) = ty {
        if let Some(seg) = type_path.path.segments.last() {
            return (seg.ident.to_string(), false);
        }
    }

    ("String".to_string(), false)
}

/// Attribute macro to convert an async function into a tool
///
/// # Example
///
/// ```rust,ignore
/// use kongfu::tool;
///
/// #[tool]
/// async fn my_tool(
///     /// The input parameter
///     input: String,
///     /// An optional count
///     count: Option<i32>
/// ) -> Result<String, Box<dyn std::error::Error>> {
///     Ok(format!("Got: {}", input))
/// }
/// ```
///
/// This generates:
/// - A `MyToolParams` struct with `#[derive(ToolParams, Deserialize)]`
/// - A `MyTool` struct implementing `ToolHandler`
#[proc_macro_attribute]
pub fn tool(args: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let args = parse_macro_input!(args as proc_macro2::TokenStream);

    let expanded = generate_tool(input, args);

    TokenStream::from(expanded)
}

/// Generate the tool implementation from an async function
fn generate_tool(func: ItemFn, args: proc_macro2::TokenStream) -> TokenStream2 {
    let func_name = &func.sig.ident;

    // Extract tool name from args or use function name
    let tool_name = extract_tool_name(&args, func_name);

    // Extract tool description from doc comments
    let tool_description = extract_function_description(&func.attrs);

    // Extract parameters
    let params = extract_parameters(&func.sig.inputs);
    let param_names: Vec<&syn::Ident> = params.iter().map(|p| &p.name).collect();

    // Generate struct names
    let params_struct_name = format_ident!("{}Params", to_pascal_case(func_name));
    let tool_struct_name = format_ident!("{}", to_pascal_case(func_name));

    // Generate the params struct fields (without doc attributes on field level)
    let params_struct_fields: Vec<TokenStream2> = params
        .iter()
        .map(|p| {
            let name = &p.name;
            let ty = &p.ty;
            quote! {
                #name: #ty
            }
        })
        .collect();

    // Generate the function body with parameter access
    let original_block = &func.block;
    let param_idents: Vec<proc_macro2::TokenStream> = params
        .iter()
        .map(|p| {
            let name = &p.name;
            quote! { params.#name }
        })
        .collect();

    // Extract the output type from Result<T, E>
    let output_type = extract_output_type(&func.sig.output);

    // Generate schema for the parameters
    let schema_properties: Vec<TokenStream2> = params
        .iter()
        .map(|p| {
            let name = &p.name;
            let name_str = name.to_string();
            let desc = &p.description;
            let ty = &p.ty;

            // Generate the JSON schema type based on the Rust type
            let json_type = get_json_type_for_type(ty);

            quote! {
                #name_str: {
                    "type": #json_type,
                    "description": #desc
                }
            }
        })
        .collect();

    let required_fields: Vec<String> = params
        .iter()
        .filter(|p| !is_optional_type(&p.ty))
        .map(|p| p.name.to_string())
        .collect();

    quote! {
        // Parameters struct
        #[derive(serde::Deserialize)]
        pub struct #params_struct_name {
            #( #params_struct_fields ),*
        }

        // Implement ToolParams for the parameters struct
        impl ::kongfu::ToolParams for #params_struct_name {
            fn schema() -> serde_json::Value {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        #( #schema_properties ),*
                    },
                    "required": [ #( #required_fields ),* ]
                })
            }

            fn from_value(value: serde_json::Value) -> std::result::Result<Self, String> {
                serde_json::from_value(value)
                    .map_err(|e| format!("Failed to parse parameters: {}", e))
            }
        }

        // Tool handler struct
        pub struct #tool_struct_name;

        #[async_trait::async_trait]
        impl ::kongfu::tools::ToolHandler for #tool_struct_name {
            type Params = #params_struct_name;
            type Output = #output_type;

            fn name(&self) -> &str {
                #tool_name
            }

            fn description(&self) -> &str {
                #tool_description
            }

            async fn execute(
                &self,
                params: Self::Params,
            ) -> ::std::result::Result<Self::Output, Box<dyn std::error::Error>> {
                // Original function body with parameter access through params
                #( let #param_names = #param_idents; )*

                #original_block
            }
        }
    }
}

/// Get the JSON schema type for a given Rust type
fn get_json_type_for_type(ty: &Type) -> String {
    let type_name = get_type_name(ty);

    match type_name.as_str() {
        "String" => "string",
        "i32" | "i64" | "u32" | "u64" => "integer",
        "f32" | "f64" => "number",
        "bool" => "boolean",
        _ => "string", // Default to string for unknown types
    }
    .to_string()
}

/// Get the type name from a Type
fn get_type_name(ty: &Type) -> String {
    if let Type::Path(type_path) = ty {
        if let Some(seg) = type_path.path.segments.last() {
            // Handle Option<T>
            if seg.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    if let Some(arg) = args.args.first() {
                        if let syn::GenericArgument::Type(inner_type) = arg {
                            return get_type_name(inner_type);
                        }
                    }
                }
            }
            return seg.ident.to_string();
        }
    }

    "String".to_string()
}

/// Check if a type is Option<T>
fn is_optional_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(seg) = type_path.path.segments.last() {
            return seg.ident == "Option";
        }
    }
    false
}

/// Extract the output type T from Result<T, E>
fn extract_output_type(return_type: &syn::ReturnType) -> TokenStream2 {
    match return_type {
        syn::ReturnType::Default => {
            quote! { () }
        }
        syn::ReturnType::Type(_, ty) => {
            // Try to extract T from Result<T, E>
            if let Type::Path(type_path) = ty.as_ref() {
                if let Some(seg) = type_path.path.segments.last() {
                    if seg.ident == "Result" {
                        if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                            if let Some(arg) = args.args.first() {
                                if let syn::GenericArgument::Type(output_type) = arg {
                                    let output_str = quote! { #output_type };
                                    return output_str;
                                }
                            }
                        }
                    }
                }
            }
            // Fallback: use the type as-is
            quote! { #ty }
        }
    }
}

/// Extract tool name from attribute args or use function name
fn extract_tool_name(args: &proc_macro2::TokenStream, func_name: &syn::Ident) -> String {
    // Try to parse name = "..." from args
    let args_str = args.to_string();
    if args_str.contains("name =") {
        // Extract the name from the args
        let parts: Vec<&str> = args_str.split('=').collect();
        if parts.len() > 1 {
            let name_part = parts[1].trim();
            // Remove quotes and trim
            let name = name_part.trim_matches('"').trim_matches('\'');
            return name.to_string();
        }
    }

    // Default to snake_case function name
    func_name.to_string()
}

/// Extract description from function doc comments
fn extract_function_description(attrs: &[Attribute]) -> String {
    let mut docs = Vec::new();

    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(nv) = &attr.meta {
                if let syn::Expr::Lit(expr) = &nv.value {
                    if let syn::Lit::Str(s) = &expr.lit {
                        let doc = s.value().trim().to_string();
                        if !doc.is_empty() {
                            docs.push(doc);
                        }
                    }
                }
            }
        }
    }

    if docs.is_empty() {
        "A tool function".to_string()
    } else {
        docs.join(" ")
    }
}

/// Parameter information extracted from function signature
struct ParameterInfo {
    name: syn::Ident,
    ty: Type,
    description: String,
}

/// Extract parameters from function inputs
fn extract_parameters(
    inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
) -> Vec<ParameterInfo> {
    inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(PatType { pat, ty, attrs, .. }) = arg {
                if let Pat::Ident(ident) = pat.as_ref() {
                    let name = ident.ident.clone();
                    let ty = ty.as_ref().clone();
                    let description = extract_param_description(attrs);
                    return Some(ParameterInfo {
                        name,
                        ty,
                        description,
                    });
                }
            }
            None
        })
        .collect()
}

/// Extract parameter description from attributes
fn extract_param_description(attrs: &[Attribute]) -> String {
    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(nv) = &attr.meta {
                if let syn::Expr::Lit(expr) = &nv.value {
                    if let syn::Lit::Str(s) = &expr.lit {
                        return s.value().trim().to_string();
                    }
                }
            }
        }
    }

    "Parameter".to_string()
}

/// Convert snake_case to PascalCase
fn to_pascal_case(ident: &syn::Ident) -> String {
    let name = ident.to_string();
    name.split('_')
        .map(|s| {
            let mut chars = s.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

/// Check if field is optional (Option<T>)
fn is_optional(field: &Field) -> bool {
    get_field_type_info(field).1
}
