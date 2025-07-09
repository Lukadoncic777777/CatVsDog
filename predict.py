from Main import Model,device,TRANSFORMER


from PIL import Image
import torch
import sys

def main():

    model=Model()
    model.to(device)
    model.load_state_dict(torch.load(sys.argv[2]))
    model.eval()
    # image=TRANSFORMER(Image.open(sys.argv[1]).convert("RGB"))
    image=Image.open(sys.argv[1]).convert('RGB')
    input_tensor=TRANSFORMER(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output,dim=1).squeeze(0)
    ans,prob='Cat',probabilities[0].item()
    if probabilities[0].item()<probabilities[1].item():
        ans,prob='Dog',probabilities[1].item()
    
    print(ans,prob)
    return 


if __name__=="__main__":
    main()