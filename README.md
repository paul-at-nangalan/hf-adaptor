# hf-adaptor
Basic Golang adaptor to run LLM models on HuggingFace using OpenAI API standard
This is still WIP, but hope it's useful to someone.

# TGI type models
##

## Example
```
var baseinstruct = `You are a ............`

func main(){

    ad := hf.NewAdaptor(hf_model_api_url, your_api_key, "tgi", baseinstruct, hf.OpenAIJsonExtractor, 12)

    somequestion := `Can you tell me how to ..........`

    answer, err := ad.SendRequest(somequestion)
    if err != nil{
        fmt.Println("ERROR: ", err)
        return
    }

    fmt.Println(answer)
}
```
