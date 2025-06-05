# hf-adaptor
A Golang adaptor for interacting with HuggingFace LLM models using an OpenAI API compatible interface.
This is still WIP, but hope it's useful to someone.

# TGI type models

### `SendRequestWithHistory`

Sends a user message to the TGI model, including the conversation history and optional tools. The 'user' role is assigned to the main message.

**Parameters:**

- `message string`: The main message content from the user.
- `history []hf.Message`: An array of previous `hf.Message` objects representing the conversation history. Each `hf.Message` object has a `Role` (e.g., "user", "assistant", "system") and `Content`.
- `tools []hf.Tool`: An optional slice of `hf.Tool` objects that the model can use.

**Return Values:**

- `string`: The textual content of the model's response.
- `[]FunctionCall`: A slice of `FunctionCall` objects if the model decides to use any of the provided tools.
- `error`: An error object if the request fails.

**Go Usage Example:**

```go
history := []hf.Message{
    {Role: "user", Content: "What was the last question I asked?"},
    {Role: "assistant", Content: "You asked about the capital of France."},
}
// Assuming 'ad' is an initialized hf.Adaptor
answer, functionCalls, err := ad.SendRequestWithHistory("What is the capital of France?", history, nil)
if err != nil {
    fmt.Println("ERROR: ", err)
    return
}
fmt.Println("Answer:", answer)
if len(functionCalls) > 0 {
    fmt.Println("Function calls:", functionCalls)
}
```

### `SendSystemRequestWithHistory`

Sends a system message to the TGI model, including the conversation history and optional tools. This is similar to `SendRequestWithHistory`, but the main message is assigned the 'system' role. System messages can be used to provide high-level instructions or context to the model.

**Parameters:**

- `message string`: The main message content (will be sent with role "system").
- `history []hf.Message`: An array of previous `hf.Message` objects.
- `tools []hf.Tool`: An optional slice of `hf.Tool` objects.

**Return Values:**

- `string`: The textual content of the model's response.
- `[]FunctionCall`: A slice of `FunctionCall` objects.
- `error`: An error object if the request fails.

**Go Usage Example:**

```go
// Assuming 'ad' is an initialized hf.Adaptor
// No history needed for this type of system message usually
responseContent, _, err := ad.SendSystemRequestWithHistory("Set the user's language to French.", []hf.Message{}, nil)
if err != nil {
    fmt.Println("ERROR: ", err)
    return
}
fmt.Println("System Response:", responseContent)
```

## Example

This example demonstrates basic usage of `NewAdaptor` and `SendRequest` for TGI models.
The `SendRequest` function is a simplified method for sending a single prompt without explicit history or tools.

```go
var baseInstruct = `You are a helpful AI assistant.` // Example base instruction

func main() {
    // Replace with your actual API URL and key
    hfModelAPIURL := "your_hf_model_api_url"
    yourAPIKey := "your_api_key"

    // hf.OpenAIJsonExtractor is an example, replace if you have a different extractor
    // The last parameter (e.g., 12) meaning might need specific documentation for the NewAdaptor function
    ad := hf.NewAdaptor(hfModelAPIURL, yourAPIKey, "tgi", baseInstruct, hf.OpenAIJsonExtractor, 12 /* e.g. MaxTokens/Timeout/SomeConfig */)

    someQuestion := `Can you tell me how to integrate this with my project?`

    answer, err := ad.SendRequest(someQuestion)
    if err != nil {
        fmt.Println("ERROR: ", err)
        return
    }

    fmt.Println("Answer:", answer)
}
```

## QnA type models

### `SendQuestion`

Sends a question along with a context to a Question and Answer (QnA) model. The model will attempt to find the answer to the question within the provided context.

**Parameters:**

- `context string`: The text containing the information where the answer should be sought.
- `question string`: The question to be answered.
- `params map[string]any`: An optional map of parameters that can be passed to the QnA model (e.g., `{"max_length": 100}`). Specific parameters depend on the model being used.

**Return Values:**

- `[]QnAResponse`: A slice of `QnAResponse` objects. Each `QnAResponse` typically includes:
    - `Answer string`: The extracted answer.
    - `Score float32`: A confidence score for the answer.
    - `Start int`: The starting character index of the answer in the context.
    - `End int`: The ending character index of the answer in the context.
- `error`: An error object if the request fails.

**Go Usage Example:**

```go
// Assuming 'qnaAd' is an initialized hf.QnAAdaptor
context := "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower."
question := "Who is the Eiffel Tower named after?"

responses, err := qnaAd.SendQuestion(context, question, nil)
if err != nil {
    fmt.Println("ERROR: ", err)
    return
}

if len(responses) > 0 {
    fmt.Println("Answer:", responses[0].Answer)
    fmt.Printf("Score: %.2f\n", responses[0].Score)
} else {
    fmt.Println("No answer found.")
}
```
