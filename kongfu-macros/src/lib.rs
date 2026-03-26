use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Data, DataStruct, DeriveInput, Field, Fields, Type, parse_macro_input};

/// Get the path to the ToolParams trait
fn get_tool_params_path() -> TokenStream2 {
    // Use crate::ToolParams - works both inside the library and in examples
    quote! { crate::ToolParams }
}

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

/// Generate JSON schema implementation
fn generate_schema_impl(input: &DeriveInput) -> TokenStream2 {
    let name = &input.ident;
    let tool_params_path = get_tool_params_path();

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

    let properties: Vec<TokenStream2> = fields
        .iter()
        .map(|field| generate_property(field))
        .collect();

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

/// Check if field is optional (Option<T>)
fn is_optional(field: &Field) -> bool {
    get_field_type_info(field).1
}
