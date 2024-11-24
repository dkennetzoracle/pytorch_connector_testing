# PyTorch Lightning Streaming

Streaming from huggingface for the optimizer is explicitly not supported (https://github.com/Lightning-AI/litdata/issues/64). You will need 300GB of storage for this step, at which point we can optimize a streaming dataset.
```
huggingface-cli download allenai/c4 --include "en/*" --local-dir /mnt/nvme/datasets/allenai/c4_en --max-workers 32 --repo-type dataset
```
this takes 4 minutes on an H100.

Then, with the local dataset, we can perform the optimizer for reading from object storage.




```
Access Key: This is the "Access Key" associated with your Customer Secret Key pair. It's provided by Oracle and associated with your Console user login.
Secret Access Key: This is the "Customer Secret Key" that you or an administrator generates to pair with the Access Key.
To obtain these credentials:
Log in to the Oracle Cloud Console.
Go to your user profile (click on the user icon in the top right corner).
Under "Resources", click on "Customer Secret Keys".
Click "Generate Secret Key".
Provide a name for the key and click "Generate Secret Key".
Copy and securely store the generated Secret Key (it will not be shown again).
The Access Key will be displayed in the list of Customer Secret Keys.
```

## Experiential Evaluations

### Dataset optimization
It is not straightforward at all to perform this.