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
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

type ToolFunctionParameters struct {
	Type                 string                                     `json:"type"` // Should be "object"
	Properties           map[string]ToolFunctionParameterProperties `json:"properties"`
	Required             []string                                   `json:"required,omitempty"`
	AdditionalProperties bool                                       `json:"additionalProperties"`
}

type Function struct {
	Name        string                  `json:"name"`
	Description string                  `json:"description,omitempty"`
	Parameters  *ToolFunctionParameters `json:"parameters"`
}
type Tool struct {
	Type     string   `json:"type"` // Should be "function"
	Function Function `json:"function"`
}

type ToolParameter struct {
	Name        string
	Type        string /// string, int ....
	Description string
	Required    bool
}

func NewTool(name string, description string, params []ToolParameter) Tool {
	tool := Tool{
		Type: "function",
	}
	function := Function{
		Name:        name,
		Description: description,
	}
	if len(params) > 0 {
		function.Parameters = &ToolFunctionParameters{
			Type:       "object",
			Properties: make(map[string]ToolFunctionParameterProperties),
		}
		required := make([]string, 0)
		for _, property := range params {
			function.Parameters.Properties[property.Name] = ToolFunctionParameterProperties{
				Type:        property.Type,
				Description: property.Description,
			}
			if property.Required {
				required = append(required, property.Name)
			}
		}
		function.Parameters.Required = required
		function.Parameters.AdditionalProperties = false
	}
	tool.Function = function
	return tool
}

type FunctionCall struct {
	Id       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Description interface{} `json:"description"`
		Name        string      `json:"name"`
		Arguments   string      `json:"arguments"`
	} `json:"function"`
}

type BaseAdaptor struct {
	apiURL     string
	apiKey     string
	model      string
	client     *http.Client
	maxretries int
}

func NewBaseAdaptor(apiurl, apikey, model string, maxretries int) *BaseAdaptor {
	return &BaseAdaptor{
		apiURL:     apiurl,
		apiKey:     apikey,
		model:      model,
		client:     &http.Client{},
		maxretries: maxretries,
	}
}

func (c *BaseAdaptor) sendWithRetry(reqData any) (*http.Response, error) {
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

// ////////////////////////////////////////////////////////////////
//
//	TGI with HUGS/OpenAI structured request response data
//
// ////////////////////////////////////////////////////////////////

type Adaptor struct {
	*BaseAdaptor
	baseinstruct string
	client       *http.Client
	extractresp  ExtractResponse
	maxretries   int
}

type ExtractResponse func(closer io.ReadCloser) (string, []FunctionCall, error)

/*
* extractresp can be nil, in which case the default extractor function (which simply extracts everything to a string)
*  will be used
* model should be the model type (which can be found somewhere on HF), e.g. tgi for text generation type models
 */
func NewAdaptor(apiurl, apikey, model string, baseinstructions string,
	extractresp ExtractResponse, maxretries int) *Adaptor {

	ad := &Adaptor{
		BaseAdaptor:  NewBaseAdaptor(apiurl, apikey, model, maxretries),
		client:       &http.Client{},
		extractresp:  extractresp,
		baseinstruct: baseinstructions,
		maxretries:   maxretries,
	}
	if extractresp == nil {
		ad.extractresp = RawExtracter
	}
	return ad
}

func (c *Adaptor) SendRequest(message string) (string, error) {
	content, _, err := c.SendRequestWithHistory(message, []Message{}, nil)
	return content, err
}

func (c *Adaptor) sendRequestWithHistory(message string, role Role, history []Message, tools []Tool) (string, []FunctionCall, error) {

	messages := make([]Message, 0, len(history)+2)

	//// The base message is instructions to the AI model
	messages = append(messages, Message{
		Role: string(ROLE_SYSTEM), Content: html.UnescapeString(c.baseinstruct),
	})
	messages = append(messages, history...)
	messages = append(messages, Message{
		Role: string(role), Content: html.UnescapeString(message),
	})
	reqData := AIRequest{
		Model:    c.model,
		Messages: messages,
	}
	if tools != nil {
		reqData.Tools = tools
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

func (c *Adaptor) SendRequestWithHistory(message string, history []Message, tools []Tool) (string, []FunctionCall, error) {
	return c.sendRequestWithHistory(message, ROLE_USER, history, tools)
}

func (c *Adaptor) SendSystemRequestWithHistory(message string, history []Message, tools []Tool) (string, []FunctionCall, error) {
	return c.sendRequestWithHistory(message, ROLE_SYSTEM, history, tools)
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
			Role      string         `json:"role"`
			Content   string         `json:"content"`
			ToolCalls []FunctionCall `json:"tool_calls,omitempty"`
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

type DebugDecoder struct {
	reader io.ReadCloser
}

func (d *DebugDecoder) Read(p []byte) (n int, err error) {
	n, err = d.reader.Read(p)
	fmt.Println(string(p))
	return n, err
}

func (d *DebugDecoder) Close() error {
	return d.reader.Close()
}

func OpenAIJsonExtractorWithDebug(reader io.ReadCloser) (string, []FunctionCall, error) {
	dbgdec := &DebugDecoder{reader: reader}

	return OpenAIJsonExtractor(dbgdec)
}

// // Extract the content field from the first message _only_
func OpenAIJsonExtractor(reader io.ReadCloser) (string, []FunctionCall, error) {
	dec := json.NewDecoder(reader)
	defer reader.Close()

	resp := Response{} // Ensure your Response struct is defined to expect FunctionCall within Message
	err := dec.Decode(&resp)
	if err != nil {
		return "", nil, err
	}
	if len(resp.Choices) > 0 {
		// Check for function call
		if resp.Choices[0].Message.ToolCalls != nil {
			return resp.Choices[0].Message.Content, resp.Choices[0].Message.ToolCalls, nil
		}
		// No function call, return content
		return resp.Choices[0].Message.Content, nil, nil
	}
	// No choices or unexpected response
	return "", nil, fmt.Errorf("no choices found in response") // Or handle as appropriate
}

func RawExtracter(reader io.ReadCloser) (string, []FunctionCall, error) {
	data, err := io.ReadAll(reader)
	if err != nil {
		return "", nil, err
	}
	fmt.Println("Resp: ", string(data))
	// RawExtracter does not parse function calls, so it returns nil for FunctionCall
	return string(data), nil, nil
}

// ///////////////////////////////////////////////////////////////////////
//
//	Question and Answer type models
//
// ///////////////////////////////////////////////////////////////////////

type QnAExtractor func(closer io.ReadCloser) ([]QnAResponse, error)

type QnAAdaptor struct {
	*BaseAdaptor

	extractor QnAExtractor
}

func NewQnAAdaptor(apiurl, apikey, model string,
	extractresp QnAExtractor, maxretries int) *QnAAdaptor {

	ad := &QnAAdaptor{
		BaseAdaptor: NewBaseAdaptor(apiurl, apikey, model, maxretries),
		extractor:   extractresp,
	}
	if extractresp == nil {
		ad.extractor = QnAJsonResponseExtractor
	}
	return ad
}

type QnAInputs struct {
	Context  string `json:"context"`  /// e.g. "My name is Clara and I live in Berkeley.",
	Question string `json:"question"` /// "What is my name?",
}
type QnARequest struct {
	Inputs     QnAInputs      `json:"inputs"`               /// "What is my name?",
	Parameters map[string]any `json:"parameters,omitempty"` //// See the model playground API in HF for these
}

func (c *QnAAdaptor) SendQuestion(context, question string, params map[string]any) ([]QnAResponse, error) {
	req := QnARequest{
		Inputs: QnAInputs{
			Context:  context,
			Question: question,
		},
		Parameters: params,
	}
	resp, err := c.sendWithRetry(req)
	handlers.PanicOnError(err)
	return c.extractor(resp.Body)
}

type QnAResponse struct {
	Answer string  `json:"answer"` //	string	The answer to the question.
	Score  float32 `json:"score"`  // number	The probability associated to the answer.
	Start  int     `json:"start"`  // The character position in the input where the answer begins.
	End    int     `json:"end"`    // The character position in the input where the answer ends
}

func QnAJsonResponseExtractorWithDebug(reader io.ReadCloser) ([]QnAResponse, error) {
	dbgreader := &DebugDecoder{reader: reader}
	return QnAJsonResponseExtractor(dbgreader)
}

func QnAJsonResponseExtractor(reader io.ReadCloser) ([]QnAResponse, error) {

	//// Response should be an array
	responses := make([]QnAResponse, 0)
	dec := json.NewDecoder(reader)
	defer reader.Close()

	err := dec.Decode(&responses)
	if err != nil {
		return nil, err
	}
	return responses, nil
}
