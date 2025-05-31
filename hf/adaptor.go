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
	Role    string `json:"role"`
	Content string `json:"content"`
}

type AIRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type ExtractResponse func(closer io.ReadCloser) (string, error)
type Adaptor struct {
	apiURL       string
	apiKey       string
	model        string
	baseinstruct string
	client       *http.Client
	extractresp  ExtractResponse
	maxretries   int
}

/*
* extractresp can be nil, in which case the default extractor function (which simply extracts everything to a string)
*  will be used
* model should be the model type (which can be found somewhere on HF), e.g. tgi for text generation type models
 */
func NewAdaptor(apiurl, apikey, model string, baseinstructions string,
	extractresp ExtractResponse, maxretries int) *Adaptor {

	ad := &Adaptor{
		apiURL:       apiurl,
		apiKey:       apikey,
		client:       &http.Client{},
		extractresp:  extractresp,
		model:        model,
		baseinstruct: baseinstructions,
		maxretries:   maxretries,
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
	return c.SendRequestWithHistory(message, []Message{})
}

func (c *Adaptor) SendRequestWithHistory(message string, history []Message) (string, error) {

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

	resp, err := c.sendWithRetry(reqData)
	handlers.PanicOnError(err)
	if resp == nil || resp.Body == nil {
		log.Panicln("Resp or resp body is nil ... this should never happen")
	}
	defer resp.Body.Close()

	return c.extractresp(resp.Body)
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
			Role    string `json:"role"`
			Content string `json:"content"`
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
func OpenAIJsonExtractor(reader io.ReadCloser) (string, error) {
	dec := json.NewDecoder(reader)
	resp := Response{}
	err := dec.Decode(&resp)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) > 0 {
		return resp.Choices[0].Message.Content, nil
	}
	return "", nil
}

func (c *Adaptor) RawExtracter(reader io.ReadCloser) (string, error) {
	data, err := io.ReadAll(reader)
	if err != nil {
		return "", err
	}
	fmt.Println("Resp: ", string(data))
	return string(data), nil
}
