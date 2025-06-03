package hf

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/paul-at-nangalan/errorhandler/handlers"
	"html"
	"io"
	"log"
	"net/http"
	"time"
)

type Role string

const (
	ROLE_SYSTEM Role = "system"
	ROLE_USER   Role = "user"
	ROLE_AGENT  Role = "assistant"
)

type Message struct {
	Role         string        `json:"role"`
	Content      string        `json:"content"` // Can be null if FunctionCall is present
	FunctionCall *FunctionCall `json:"function_call,omitempty"`
}

type AIRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
}

type ToolFunctionParameterProperties struct {
	Type        string   `json:"type"`
	Description string   `json:"description,omitempty"`
	Enum        []string `json:"enum,omitempty"`
}

type ToolFunctionParameters struct {
	Type                 string                                     `json:"type"` // Should be "object"
	Properties           map[string]ToolFunctionParameterProperties `json:"properties"`
	Required             []string                                   `json:"required,omitempty"`
	AdditionalProperties bool                                       `json:"additionalProperties"`
}

type ToolFunction struct {
	Name        string                  `json:"name"`
	Description string                  `json:"description,omitempty"`
	Parameters  *ToolFunctionParameters `json:"parameters"`
}

type Tool struct {
	Type     string       `json:"type"` // Should be "function"
	Function ToolFunction `json:"function"`
}

type FunctionCall struct {
	ID        string `json:"id,omitempty"`      // Optional, as per user's example
	CallID    string `json:"call_id,omitempty"` // Optional, as per user's example
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // This is a JSON string
}

type RegisteredFunction func(params map[string]any, hiddenParams map[string]any) (string, error)

type ExtractResponse func(closer io.ReadCloser) (string, *FunctionCall, error)
type Adaptor struct {
	apiURL              string
	apiKey              string
	model               string
	baseinstruct        string
	client              *http.Client
	extractresp         ExtractResponse
	maxretries          int
	registeredFunctions map[string]RegisteredFunction // New field
	tools               []Tool                        // New field
}

/*
* extractresp can be nil, in which case the default extractor function (which simply extracts everything to a string)
*  will be used
* model should be the model type (which can be found somewhere on HF), e.g. tgi for text generation type models
 */
func NewAdaptor(apiurl, apikey, model string, baseinstructions string,
	extractresp ExtractResponse, maxretries int,
	userFunctions map[string]RegisteredFunction, userTools []Tool) *Adaptor {

	ad := &Adaptor{
		apiURL:              apiurl,
		apiKey:              apikey,
		client:              &http.Client{},
		extractresp:         extractresp,
		model:               model,
		baseinstruct:        baseinstructions,
		maxretries:          maxretries,
		registeredFunctions: make(map[string]RegisteredFunction), // Initialize map
		tools:               make([]Tool, 0),                     // Initialize slice
	}
	if userFunctions != nil {
		ad.registeredFunctions = userFunctions
	}
	if userTools != nil {
		ad.tools = userTools
	}
	if extractresp == nil {
		ad.extractresp = ad.RawExtracter
	}
	return ad
}

func (c *Adaptor) sendWithRetry(reqData any) (*http.Response, error) {
	for i := 0; i < c.maxretries; i++ {
		body := &bytes.Buffer{}
		err := json.NewEncoder(body).Encode(reqData)
		handlers.PanicOnError(err)

		//fmt.Println("Calling agent with ", c.apiURL, " and key ", c.apiKey)
		req, err := http.NewRequest(http.MethodPost, c.apiURL, body)
		if err != nil {
			return nil, fmt.Errorf("error creating request: %w", err)
		}

		req.Header.Set("Accept", "application/json")
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+c.apiKey)

		resp, err := c.client.Do(req)

		if err != nil {
			return nil, fmt.Errorf("error sending request: %w", err)
		}
		/// retry
		if resp.StatusCode == 503 {
			fmt.Println("Status code 503 - service not ready - sleeping for 30 seconds with max ", c.maxretries, " retries")
			resp.Body.Close()
			time.Sleep(30 * time.Second)
			continue
		}
		if resp.StatusCode != http.StatusOK {
			errmsg, err := io.ReadAll(resp.Body)
			log.Println("Error: ", string(errmsg), " err ", err)
			if resp.Body != nil {
				resp.Body.Close()
			}
			return nil, fmt.Errorf("API request failed with status %d", resp.StatusCode)
		}

		return resp, nil
	}
	return nil, fmt.Errorf("Num retries exceeded")
}

func (c *Adaptor) SendRequest(message string) (string, error) {
	content, _, err := c.SendRequestWithHistory(message, []Message{})
	return content, err
}

func (c *Adaptor) SendRequestWithHistory(message string, history []Message) (string, *FunctionCall, error) {

	messages := make([]Message, 0, len(history)+2)

	messages = append(messages, Message{
		Role: string(ROLE_SYSTEM), Content: html.UnescapeString(c.baseinstruct),
	})
	messages = append(messages, history...)
	messages = append(messages, Message{
		Role: string(ROLE_USER), Content: html.UnescapeString(message),
	})
	reqData := AIRequest{
		Model:    c.model,
		Messages: messages,
	}
	if len(c.tools) > 0 {
		reqData.Tools = c.tools
	}

	resp, err := c.sendWithRetry(reqData)
	handlers.PanicOnError(err)
	if resp == nil || resp.Body == nil {
		log.Panicln("Resp or resp body is nil ... this should never happen")
	}
	defer resp.Body.Close()

	content, functionCall, err := c.extractresp(resp.Body)
	return content, functionCall, err
}

type Response struct {
	Object            string `json:"object"`
	Id                string `json:"id"`
	Created           int    `json:"created"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index   int `json:"index"`
		Message struct {
			Role         string        `json:"role"`
			Content      string        `json:"content"`
			FunctionCall *FunctionCall `json:"function_call,omitempty"`
		} `json:"message"`
		Logprobs     interface{} `json:"logprobs"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// // Extract the content field from the first message _only_
func OpenAIJsonExtractor(reader io.ReadCloser) (string, *FunctionCall, error) {
	dec := json.NewDecoder(reader)
	resp := Response{} // Ensure your Response struct is defined to expect FunctionCall within Message
	err := dec.Decode(&resp)
	if err != nil {
		return "", nil, err
	}
	if len(resp.Choices) > 0 {
		// Check for function call
		if resp.Choices[0].Message.FunctionCall != nil {
			return resp.Choices[0].Message.Content, resp.Choices[0].Message.FunctionCall, nil
		}
		// No function call, return content
		return resp.Choices[0].Message.Content, nil, nil
	}
	// No choices or unexpected response
	return "", nil, fmt.Errorf("no choices found in response") // Or handle as appropriate
}

func (c *Adaptor) RawExtracter(reader io.ReadCloser) (string, *FunctionCall, error) {
	data, err := io.ReadAll(reader)
	if err != nil {
		return "", nil, err
	}
	fmt.Println("Resp: ", string(data))
	// RawExtracter does not parse function calls, so it returns nil for FunctionCall
	return string(data), nil, nil
}
