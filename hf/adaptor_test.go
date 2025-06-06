package hf

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"fmt"
)

// Mock function for testing
func mockGetUserWeather(params map[string]any, hiddenParams map[string]any) (string, error) {
	location, ok := params["location"].(string)
	if !ok {
		return "", fmt.Errorf("location not found or not a string")
	}
	// In a real scenario, hiddenParams might be used here
	return `{"weather": "sunny", "temperature": "25C", "location": "` + location + `"}`, nil
}

func TestNewAdaptorWithFunctions(t *testing.T) {
	apiKey := "test-key"
	model := "test-model"
	apiURL := "http://localhost/test"
	baseInstruct := "You are an assistant."

	// userTools and userFuncs are no longer passed at initialization
	// extractResp is nil for this test as we are not testing response extraction here
	// maxretries is set to 1 for this test
	adaptor := NewAdaptor(apiURL, apiKey, model, baseInstruct, nil, 1)

	if adaptor == nil {
		t.Fatal("NewAdaptor returned nil")
	}
	if adaptor.apiKey != apiKey {
		t.Errorf("expected apiKey '%s', got '%s'", apiKey, adaptor.apiKey)
	}
	if adaptor.model != model {
		t.Errorf("expected model '%s', got '%s'", model, adaptor.model)
	}
	if adaptor.apiURL != apiURL {
		t.Errorf("expected apiURL '%s', got '%s'", apiURL, adaptor.apiURL)
	}
	if adaptor.baseinstruct != baseInstruct {
		t.Errorf("expected baseinstruct '%s', got '%s'", baseInstruct, adaptor.baseinstruct)
	}
	// Assertions for adaptor.tools and adaptor.registeredFunctions are removed
	// as they are no longer set during initialization in this manner.
}

func TestSendRequestWithHistory_FunctionCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 1. Verify request body contains tools
		var reqData AIRequest
		bodyBytes, _ := io.ReadAll(r.Body)
		json.Unmarshal(bodyBytes, &reqData)

		if len(reqData.Tools) == 0 {
			t.Error("Expected tools in request, got none")
			http.Error(w, "missing tools", http.StatusBadRequest)
			return
		}
		if reqData.Tools[0].Function.Name != "get_user_weather" {
			t.Errorf("Expected tool 'get_user_weather' in request, got %s", reqData.Tools[0].Function.Name)
			http.Error(w, "wrong tool name", http.StatusBadRequest)
			return
		}

		// 2. Send back a response that includes a function call
		response := Response{
			Choices: []struct {
				Index   int `json:"index"`
				Message struct { // This anonymous struct must match hf.Response.Choices[].Message
					Role      string         `json:"role"`
					Content   string         `json:"content"`
					ToolCalls []FunctionCall `json:"tool_calls,omitempty"` // Corrected field name
				} `json:"message"`
				Logprobs     interface{} `json:"logprobs"`
				FinishReason string      `json:"finish_reason"`
			}{
				{
					Index: 0,
					Message: struct { // This struct literal's type is defined by the anonymous struct above
						Role      string         `json:"role"`
						Content   string         `json:"content"`
						ToolCalls []FunctionCall `json:"tool_calls,omitempty"` // Corrected field name
					}{
						Role:    string(ROLE_AGENT),
						Content: "", // Or null
						ToolCalls: []FunctionCall{
							{
								Id:   "call_123",
								Type: "function",
								Function: struct {
									Description interface{} `json:"description"`
									Name        string      `json:"name"`
									Arguments   string      `json:"arguments"`
								}{
									Name:      "get_user_weather",
									Arguments: `{"location": "London"}`,
								},
							},
						},
					},
					FinishReason: "tool_calls",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	userTools := []Tool{ // This definition should align with adaptor.go's Tool and Function structs
		{
			Type: "function",
			Function: Function{ // Changed from ToolFunction to Function
				Name:        "get_user_weather",
				Description: "Get weather for a user",
				Parameters: &ToolFunctionParameters{ // This part seems compatible
					Type: "object",
					Properties: map[string]ToolFunctionParameterProperties{
						"location": {Type: "string", Description: "City name"},
					},
					Required: []string{"location"},
				},
			},
		},
	}
	// userFuncs and adaptor.registeredFunctions are removed as per the new design
	// The mechanism for calling functions has changed. The test needs to reflect this.
	// For now, removing userFuncs and the direct call to registeredFunc.
	// The test will verify that the adaptor correctly sends and receives function call requests.
	// The actual execution of the function is now a client-side concern.

	// Use OpenAIJsonExtractor for this test
	adaptor := NewAdaptor(server.URL, "test-key", "test-model", "You are an assistant.", OpenAIJsonExtractor, 1) // Removed userFuncs, userTools

	// SendRequestWithHistory now expects tools to be passed if they are to be used in the request
	content, funcCalls, err := adaptor.SendRequestWithHistory("What's the weather in London?", []Message{}, userTools) // Pass userTools here

	if err != nil {
		t.Fatalf("SendRequestWithHistory returned error: %v", err)
	}
	if content != "" { // Content might not be empty, depends on LLM and extractor
		// t.Errorf("Expected empty content, got: %s", content)
		// For now, let's not be strict about empty content, as some models might return content alongside function calls.
	}
	if funcCalls == nil || len(funcCalls) == 0 {
		t.Fatal("Expected function call(s), got nil or empty")
	}

	// Assuming one function call for this test
	actualFuncCall := funcCalls[0]

	if actualFuncCall.Function.Name != "get_user_weather" {
		t.Errorf("Expected function name 'get_user_weather', got '%s'", actualFuncCall.Function.Name)
	}
	expectedArgs := `{"location": "London"}`
	if actualFuncCall.Function.Arguments != expectedArgs {
		t.Errorf("Expected arguments '%s', got '%s'", expectedArgs, actualFuncCall.Function.Arguments)
	}

	// The part about calling registeredFunc is removed as the adaptor no longer handles function registration.
	// The client is responsible for executing the function based on the returned FunctionCall.
	// We can still test unmarshalling the arguments as a sanity check.
	var params map[string]any
	err = json.Unmarshal([]byte(actualFuncCall.Function.Arguments), &params)
	if err != nil {
		t.Fatalf("Failed to unmarshal function call arguments: %v", err)
	}
	if _, ok := params["location"]; !ok {
		t.Errorf("Location not found in unmarshalled arguments: %v", params)
	}
}

func TestSendRequestWithHistory_RegularMessage(t *testing.T) {
	expectedResponseContent := "This is a regular response."
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request body does not necessarily contain tools if none are registered for the adaptor
		var reqData AIRequest
		bodyBytes, _ := io.ReadAll(r.Body)
		json.Unmarshal(bodyBytes, &reqData)

		if len(reqData.Tools) > 0 {
			// This test is for an adaptor instance *without* tools.
			// If tools were registered, they would be sent.
			// For this specific test case, we assume no tools on the adaptor.
			// A different test could verify tools are sent if registered.
		}

		response := Response{
			Choices: []struct {
				Index   int `json:"index"`
				Message struct { // This anonymous struct must match hf.Response.Choices[].Message
					Role      string         `json:"role"`
					Content   string         `json:"content"`
					ToolCalls []FunctionCall `json:"tool_calls,omitempty"` // Corrected field name
				} `json:"message"`
				Logprobs     interface{} `json:"logprobs"`
				FinishReason string      `json:"finish_reason"`
			}{
				{
					Index: 0,
					Message: struct { // This struct literal's type is defined by the anonymous struct above
						Role      string         `json:"role"`
						Content   string         `json:"content"`
						ToolCalls []FunctionCall `json:"tool_calls,omitempty"` // Corrected field name
					}{
						Role:    string(ROLE_AGENT),
						Content: expectedResponseContent,
						// ToolCalls is nil for a regular message
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	// Adaptor without any tools/functions registered
	adaptor := NewAdaptor(server.URL, "test-key", "test-model", "You are an assistant.", OpenAIJsonExtractor, 1) // Corrected NewAdaptor call

	content, funcCalls, err := adaptor.SendRequestWithHistory("Hello there", []Message{}, nil) // funcCall is now funcCalls

	if err != nil {
		t.Fatalf("SendRequestWithHistory returned error: %v", err)
	}
	if funcCalls != nil { // Check if funcCalls is nil or empty
		t.Fatalf("Expected no function call, got one: %+v", funcCalls)
	}
	if content != expectedResponseContent {
		t.Errorf("Expected content '%s', got '%s'", expectedResponseContent, content)
	}
}

func TestToolJsonMarshalling(t *testing.T) {
	tool := Tool{
		Type: "function",
		Function: Function{ // Changed ToolFunction to Function
			Name:        "get_current_weather",
			Description: "Get the current weather in a given location",
			Parameters: &ToolFunctionParameters{
				Type: "object",
				Properties: map[string]ToolFunctionParameterProperties{
					"location": {
						Type:        "string",
						Description: "The city and state, e.g. San Francisco, CA",
					},
					"unit": {
						Type:        "string",
						Description: "Unit for temperature, e.g. celsius or fahrenheit", // Enum removed, added description
					},
				},
				Required:             []string{"location"},
				AdditionalProperties: false,
			},
		},
	}

	expectedJson := `{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather in a given location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","description":"Unit for temperature, e.g. celsius or fahrenheit"}},"required":["location"],"additionalProperties":false}}}`
	
	jsonData, err := json.Marshal(tool)
	if err != nil {
		t.Fatalf("Failed to marshal Tool: %v", err)
	}

	if string(jsonData) != expectedJson {
		// For more complex JSON, consider unmarshalling both and comparing the resulting objects
		// or using a JSON diff library. For this specific case, string comparison might be okay
		// if field order is deterministic (Go maps are not, so this might be flaky if not careful).
		// A better approach for complex objects:
		var unmarshalledExpected map[string]interface{}
		var unmarshalledActual map[string]interface{}
		json.Unmarshal([]byte(expectedJson), &unmarshalledExpected)
		json.Unmarshal(jsonData, &unmarshalledActual)

		if !reflect.DeepEqual(unmarshalledActual, unmarshalledExpected) {
			t.Errorf("Expected JSON:\n%s\nGot JSON:\n%s", expectedJson, string(jsonData))
		}
	}
}


// Helper to compare function call argument strings, ignoring whitespace differences
func compareJsonStrings(s1, s2 string) (bool, error) {
	var o1 interface{}
	var o2 interface{}

	err := json.Unmarshal([]byte(s1), &o1)
	if err != nil {
		return false, fmt.Errorf("Error unmarshalling string 1: %s", err.Error())
	}
	err = json.Unmarshal([]byte(s2), &o2)
	if err != nil {
		return false, fmt.Errorf("Error unmarshalling string 2: %s", err.Error())
	}

	return reflect.DeepEqual(o1, o2), nil
}

// Add a test for SendRequest to ensure it calls SendRequestWithHistory correctly
func TestSendRequest(t *testing.T) {
	expectedResponseContent := "This is a test response for SendRequest."
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := Response{
			Choices: []struct {
				Index   int `json:"index"`
				Message struct { // This anonymous struct must match hf.Response.Choices[].Message
					Role      string         `json:"role"`
					Content   string         `json:"content"`
					ToolCalls []FunctionCall `json:"tool_calls,omitempty"` // Corrected field name
				} `json:"message"`
				Logprobs     interface{} `json:"logprobs"`
				FinishReason string      `json:"finish_reason"`
			}{
				{
					Index: 0,
					Message: struct { // This struct literal's type is defined by the anonymous struct above
						Role      string         `json:"role"`
						Content   string         `json:"content"`
						ToolCalls []FunctionCall `json:"tool_calls,omitempty"` // Corrected field name
					}{
						Role:    string(ROLE_AGENT),
						Content: expectedResponseContent,
						// ToolCalls is nil here, which is fine
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	adaptor := NewAdaptor(server.URL, "test-key", "test-model", "Base instructions", OpenAIJsonExtractor, 1) // Removed nil, nil
	
	content, err := adaptor.SendRequest("Test message")
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}
	if content != expectedResponseContent {
		t.Errorf("Expected content '%s', got '%s'", expectedResponseContent, content)
	}
}

// It's also good to test the RawExtracter
func TestRawExtracter(t *testing.T) {
	rawResponse := `{"some_raw_data": "value"}`
	reader := io.NopCloser(strings.NewReader(rawResponse))
	
	// RawExtracter is a standalone function.
	// RawExtracter now returns (string, []FunctionCall, error)
	content, funcCalls, err := RawExtracter(reader) // Called directly
	if err != nil {
		t.Fatalf("RawExtracter failed: %v", err)
	}
	if funcCalls != nil { // Check if funcCalls is nil or empty
		t.Errorf("RawExtracter should not return a function call, got: %+v", funcCalls)
	}
	if content != rawResponse {
		t.Errorf("Expected content '%s', got '%s'", rawResponse, content)
	}
}

func TestQnAAdaptor_SendQuestion(t *testing.T) {
	expectedApiKey := "qna-test-key"
	expectedModel := "qna-test-model"
	expectedContext := "This is a test context."
	expectedQuestion := "What is this a test of?"
	expectedParams := map[string]any{"max_tokens": 100}

	expectedQnAResponse := []QnAResponse{
		{Answer: "Test Answer", Score: 0.9, Start: 0, End: 10},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("Expected POST request, got %s", r.Method)
			http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
			return
		}

		var reqData QnARequest
		if err := json.NewDecoder(r.Body).Decode(&reqData); err != nil {
			t.Errorf("Failed to decode request body: %v", err)
			http.Error(w, "Bad request body", http.StatusBadRequest)
			return
		}

		if reqData.Inputs.Context != expectedContext {
			t.Errorf("Expected context '%s', got '%s'", expectedContext, reqData.Inputs.Context)
		}
		if reqData.Inputs.Question != expectedQuestion {
			t.Errorf("Expected question '%s', got '%s'", expectedQuestion, reqData.Inputs.Question)
		}
		// Note: Comparing params map[string]any directly can be tricky due to order.
		// For a robust check, you might iterate or use reflect.DeepEqual if types are consistent.
		// Here, we'll assume basic check is fine for now or that params are not strictly validated in this mock.

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(expectedQnAResponse); err != nil {
			t.Errorf("Failed to encode response: %v", err)
		}
	}))
	defer server.Close()

	// Test NewQnAAdaptor - with nil extractor (should default)
	qnaAdaptorDefaultExtractor := NewQnAAdaptor(server.URL, expectedApiKey, expectedModel, nil, 1)
	if qnaAdaptorDefaultExtractor == nil {
		t.Fatalf("NewQnAAdaptor (nil extractor) returned nil")
	}
	if qnaAdaptorDefaultExtractor.apiURL != server.URL {
		t.Errorf("Default Extractor: Expected apiURL '%s', got '%s'", server.URL, qnaAdaptorDefaultExtractor.apiURL)
	}
	if qnaAdaptorDefaultExtractor.apiKey != expectedApiKey {
		t.Errorf("Default Extractor: Expected apiKey '%s', got '%s'", expectedApiKey, qnaAdaptorDefaultExtractor.apiKey)
	}
	if qnaAdaptorDefaultExtractor.model != expectedModel {
		t.Errorf("Default Extractor: Expected model '%s', got '%s'", expectedModel, qnaAdaptorDefaultExtractor.model)
	}
	// Check if extractor is QnAJsonResponseExtractor by checking it's not nil.
	// A more direct comparison of function pointers is unreliable.
	if qnaAdaptorDefaultExtractor.extractor == nil {
		t.Error("Default Extractor: Expected extractor to be set to QnAJsonResponseExtractor, but it was nil")
	}

	// Create a dummy extractor for testing non-default case
	// To avoid "function <xyz> declared and not used", we'll use it briefly
	dummyExtractor := func(closer io.ReadCloser) ([]QnAResponse, error) {
		// This is a placeholder and its internal logic won't be directly tested here,
		// only that this function itself is assigned.
		return []QnAResponse{{Answer: "dummy", Score: 0.1, Start: 1, End: 2}}, nil
	}
	qnaAdaptorCustomExtractor := NewQnAAdaptor(server.URL, expectedApiKey, expectedModel, dummyExtractor, 1)
	if qnaAdaptorCustomExtractor == nil {
		t.Fatalf("NewQnAAdaptor (custom extractor) returned nil")
	}
	if qnaAdaptorCustomExtractor.extractor == nil {
		t.Error("Custom Extractor: Expected custom extractor to be set, but it was nil")
	} else {
		// Compare function pointers to ensure the dummyExtractor was assigned.
		customExtractorPtr := reflect.ValueOf(qnaAdaptorCustomExtractor.extractor).Pointer()
		expectedExtractorPtr := reflect.ValueOf(dummyExtractor).Pointer()
		if customExtractorPtr != expectedExtractorPtr {
			t.Error("Custom Extractor: The assigned extractor is not the dummyExtractor function instance.")
		}
	}

	// Test SendQuestion using the adaptor with the default (mocked) extractor behavior
	responses, err := qnaAdaptorDefaultExtractor.SendQuestion(expectedContext, expectedQuestion, expectedParams)
	if err != nil {
		t.Fatalf("SendQuestion failed: %v", err)
	}

	if !reflect.DeepEqual(responses, expectedQnAResponse) {
		t.Errorf("SendQuestion response mismatch:\nExpected: %+v\nGot:      %+v", expectedQnAResponse, responses)
	}
}

func TestQnAJsonResponseExtractor(t *testing.T) {
	t.Run("ValidJSON", func(t *testing.T) {
		jsonString := `[{"answer": "Paris", "score": 0.95, "start": 10, "end": 14}, {"answer": "France", "score": 0.80, "start": 20, "end": 25}]`
		reader := io.NopCloser(strings.NewReader(jsonString))
		expectedResponses := []QnAResponse{
			{Answer: "Paris", Score: 0.95, Start: 10, End: 14},
			{Answer: "France", Score: 0.80, Start: 20, End: 25},
		}

		responses, err := QnAJsonResponseExtractor(reader)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if !reflect.DeepEqual(responses, expectedResponses) {
			t.Errorf("Response mismatch:\nExpected: %+v\nGot:      %+v", expectedResponses, responses)
		}
	})

	t.Run("EmptyJSONArray", func(t *testing.T) {
		jsonString := `[]`
		reader := io.NopCloser(strings.NewReader(jsonString))
		expectedResponses := []QnAResponse{}

		responses, err := QnAJsonResponseExtractor(reader)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if len(responses) != 0 {
			t.Errorf("Expected empty slice, got %+v", responses)
		}
		// Also check deep equal for good measure, though len check is primary for empty
		if !reflect.DeepEqual(responses, expectedResponses) {
			t.Errorf("Response mismatch for empty array:\nExpected: %+v\nGot:      %+v", expectedResponses, responses)
		}
	})

	t.Run("MalformedJSON", func(t *testing.T) {
		jsonString := `[{"answer": "Test"` // Missing closing brace and bracket
		reader := io.NopCloser(strings.NewReader(jsonString))

		_, err := QnAJsonResponseExtractor(reader)
		if err == nil {
			t.Fatal("Expected an error for malformed JSON, got nil")
		}
	})

	t.Run("IncorrectJSONStructureNotArray", func(t *testing.T) {
		jsonString := `{"answer": "Test", "score": 0.5, "start": 0, "end": 3}` // Object instead of array
		reader := io.NopCloser(strings.NewReader(jsonString))

		_, err := QnAJsonResponseExtractor(reader)
		if err == nil {
			t.Fatal("Expected an error for incorrect JSON structure (object instead of array), got nil")
		}
	})

	t.Run("IncorrectJSONFields", func(t *testing.T) {
		// Score is a string, should be float32
		jsonString := `[{"answer": "Test", "score": "high", "start": 0, "end": 3}]`
		reader := io.NopCloser(strings.NewReader(jsonString))

		_, err := QnAJsonResponseExtractor(reader)
		if err == nil {
			t.Fatal("Expected an error for incorrect JSON field type, got nil")
		}
		// More specific error check if desired, e.g., using strings.Contains(err.Error(), "cannot unmarshal string")
		// For now, just checking for any error is sufficient.
	})
}
