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

	userTools := []Tool{
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "get_user_weather",
				Description: "Get weather for a user",
				Parameters: &ToolFunctionParameters{
					Type: "object",
					Properties: map[string]ToolFunctionParameterProperties{
						"location": {Type: "string", Description: "City name"},
					},
					Required: []string{"location"},
				},
			},
		},
	}
	userFuncs := map[string]RegisteredFunction{
		"get_user_weather": mockGetUserWeather,
	}

	adaptor := NewAdaptor(apiURL, apiKey, model, baseInstruct, nil, 1, userFuncs, userTools)

	if adaptor == nil {
		t.Fatal("NewAdaptor returned nil")
	}
	if adaptor.apiKey != apiKey {
		t.Errorf("expected apiKey %s, got %s", apiKey, adaptor.apiKey)
	}
	if len(adaptor.tools) != 1 {
		t.Errorf("expected 1 tool, got %d", len(adaptor.tools))
	} else {
		if adaptor.tools[0].Function.Name != "get_user_weather" {
			t.Errorf("expected tool name 'get_user_weather', got '%s'", adaptor.tools[0].Function.Name)
		}
	}
	if len(adaptor.registeredFunctions) != 1 {
		t.Errorf("expected 1 registered function, got %d", len(adaptor.registeredFunctions))
	}
	if _, ok := adaptor.registeredFunctions["get_user_weather"]; !ok {
		t.Errorf("expected 'get_user_weather' function to be registered")
	}
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
				Message struct {
					Role         string          `json:"role"`
					Content      string          `json:"content"`
					FunctionCall *FunctionCall   `json:"function_call,omitempty"`
				} `json:"message"`
				Logprobs     interface{} `json:"logprobs"`
				FinishReason string      `json:"finish_reason"`
			}{
				{
					Index: 0,
					Message: struct {
						Role         string          `json:"role"`
						Content      string          `json:"content"`
						FunctionCall *FunctionCall   `json:"function_call,omitempty"`
					}{
						Role:    string(ROLE_AGENT),
						Content: "", // Or null
						FunctionCall: &FunctionCall{
							Name:      "get_user_weather",
							Arguments: `{"location": "London"}`,
						},
					},
					FinishReason: "function_call",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	userTools := []Tool{
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "get_user_weather",
				Description: "Get weather for a user",
				Parameters: &ToolFunctionParameters{
					Type: "object",
					Properties: map[string]ToolFunctionParameterProperties{
						"location": {Type: "string", Description: "City name"},
					},
					Required: []string{"location"},
				},
			},
		},
	}
	userFuncs := map[string]RegisteredFunction{
		"get_user_weather": mockGetUserWeather,
	}

	// Use OpenAIJsonExtractor for this test
	adaptor := NewAdaptor(server.URL, "test-key", "test-model", "You are an assistant.", OpenAIJsonExtractor, 1, userFuncs, userTools)

	content, funcCall, err := adaptor.SendRequestWithHistory("What's the weather in London?", []Message{}, nil)

	if err != nil {
		t.Fatalf("SendRequestWithHistory returned error: %v", err)
	}
	if content != "" { // Expect empty content when a function call is made
		t.Errorf("Expected empty content, got: %s", content)
	}
	if funcCall == nil {
		t.Fatal("Expected function call, got nil")
	}
	if funcCall.Name != "get_user_weather" {
		t.Errorf("Expected function name 'get_user_weather', got '%s'", funcCall.Name)
	}
	expectedArgs := `{"location": "London"}`
	if funcCall.Arguments != expectedArgs {
		t.Errorf("Expected arguments '%s', got '%s'", expectedArgs, funcCall.Arguments)
	}

	// Verify that the registered function can be called with the arguments (simulating client behavior)
	var params map[string]any
	err = json.Unmarshal([]byte(funcCall.Arguments), &params)
	if err != nil {
		t.Fatalf("Failed to unmarshal function call arguments: %v", err)
	}
	
	registeredFunc, ok := adaptor.registeredFunctions[funcCall.Name]
	if !ok {
		t.Fatalf("Function %s not found in registered functions", funcCall.Name)
	}
	
	_, err = registeredFunc(params, nil) // Pass nil for hiddenParams for this test
	if err != nil {
		t.Fatalf("Registered function execution failed: %v", err)
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
				Message struct {
					Role         string          `json:"role"`
					Content      string          `json:"content"`
					FunctionCall *FunctionCall   `json:"function_call,omitempty"`
				} `json:"message"`
				Logprobs     interface{} `json:"logprobs"`
				FinishReason string      `json:"finish_reason"`
			}{
				{
					Index: 0,
					Message: struct {
						Role         string          `json:"role"`
						Content      string          `json:"content"`
						FunctionCall *FunctionCall   `json:"function_call,omitempty"`
					}{
						Role:    string(ROLE_AGENT),
						Content: expectedResponseContent,
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
	adaptor := NewAdaptor(server.URL, "test-key", "test-model", "You are an assistant.", OpenAIJsonExtractor, 1, nil, nil)

	content, funcCall, err := adaptor.SendRequestWithHistory("Hello there", []Message{}, nil)

	if err != nil {
		t.Fatalf("SendRequestWithHistory returned error: %v", err)
	}
	if funcCall != nil {
		t.Fatalf("Expected no function call, got one: %+v", funcCall)
	}
	if content != expectedResponseContent {
		t.Errorf("Expected content '%s', got '%s'", expectedResponseContent, content)
	}
}

func TestToolJsonMarshalling(t *testing.T) {
	tool := Tool{
		Type: "function",
		Function: ToolFunction{
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
						Type: "string",
						Enum: []string{"celsius", "fahrenheit"},
					},
				},
				Required:             []string{"location"},
				AdditionalProperties: false,
			},
		},
	}

	expectedJson := `{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather in a given location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"],"additionalProperties":false}}}`
	
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
				Message struct {
					Role         string          `json:"role"`
					Content      string          `json:"content"`
					FunctionCall *FunctionCall   `json:"function_call,omitempty"`
				} `json:"message"`
				Logprobs     interface{} `json:"logprobs"`
				FinishReason string      `json:"finish_reason"`
			}{
				{
					Index: 0,
					Message: struct {
						Role         string          `json:"role"`
						Content      string          `json:"content"`
						FunctionCall *FunctionCall   `json:"function_call,omitempty"`
					}{
						Role:    string(ROLE_AGENT),
						Content: expectedResponseContent,
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	adaptor := NewAdaptor(server.URL, "test-key", "test-model", "Base instructions", OpenAIJsonExtractor, 1, nil, nil)
	
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
	
	// Instantiate a dummy adaptor to call RawExtracter
	adaptor := &Adaptor{} 
	content, funcCall, err := adaptor.RawExtracter(reader)
	if err != nil {
		t.Fatalf("RawExtracter failed: %v", err)
	}
	if funcCall != nil {
		t.Errorf("RawExtracter should not return a function call, got: %+v", funcCall)
	}
	if content != rawResponse {
		t.Errorf("Expected content '%s', got '%s'", rawResponse, content)
	}
}
