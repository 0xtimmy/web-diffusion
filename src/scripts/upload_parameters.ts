import * as torch from "@/torch"
const params = ["bot1.double_conv.0.weight", "bot1.double_conv.1.bias", "bot1.double_conv.1.weight", "bot1.double_conv.3.weight", "bot1.double_conv.4.bias", "bot1.double_conv.4.weight", "bot2.double_conv.0.weight", "bot2.double_conv.1.bias", "bot2.double_conv.1.weight", "bot2.double_conv.3.weight", "bot2.double_conv.4.bias", "bot2.double_conv.4.weight", "bot3.double_conv.0.weight", "bot3.double_conv.1.bias", "bot3.double_conv.1.weight", "bot3.double_conv.3.weight", "bot3.double_conv.4.bias", "bot3.double_conv.4.weight", "down1.emb_layer.1.bias", "down1.emb_layer.1.weight", "down1.maxpool_conv.1.double_conv.0.weight", "down1.maxpool_conv.1.double_conv.1.bias", "down1.maxpool_conv.1.double_conv.1.weight", "down1.maxpool_conv.1.double_conv.3.weight", "down1.maxpool_conv.1.double_conv.4.bias", "down1.maxpool_conv.1.double_conv.4.weight", "down1.maxpool_conv.2.double_conv.0.weight", "down1.maxpool_conv.2.double_conv.1.bias", "down1.maxpool_conv.2.double_conv.1.weight", "down1.maxpool_conv.2.double_conv.3.weight", "down1.maxpool_conv.2.double_conv.4.bias", "down1.maxpool_conv.2.double_conv.4.weight", "down2.emb_layer.1.bias", "down2.emb_layer.1.weight", "down2.maxpool_conv.1.double_conv.0.weight", "down2.maxpool_conv.1.double_conv.1.bias", "down2.maxpool_conv.1.double_conv.1.weight", "down2.maxpool_conv.1.double_conv.3.weight", "down2.maxpool_conv.1.double_conv.4.bias", "down2.maxpool_conv.1.double_conv.4.weight", "down2.maxpool_conv.2.double_conv.0.weight", "down2.maxpool_conv.2.double_conv.1.bias", "down2.maxpool_conv.2.double_conv.1.weight", "down2.maxpool_conv.2.double_conv.3.weight", "down2.maxpool_conv.2.double_conv.4.bias", "down2.maxpool_conv.2.double_conv.4.weight", "down3.emb_layer.1.bias", "down3.emb_layer.1.weight", "down3.maxpool_conv.1.double_conv.0.weight", "down3.maxpool_conv.1.double_conv.1.bias", "down3.maxpool_conv.1.double_conv.1.weight", "down3.maxpool_conv.1.double_conv.3.weight", "down3.maxpool_conv.1.double_conv.4.bias", "down3.maxpool_conv.1.double_conv.4.weight", "down3.maxpool_conv.2.double_conv.0.weight", "down3.maxpool_conv.2.double_conv.1.bias", "down3.maxpool_conv.2.double_conv.1.weight", "down3.maxpool_conv.2.double_conv.3.weight", "down3.maxpool_conv.2.double_conv.4.bias", "down3.maxpool_conv.2.double_conv.4.weight", "inc.double_conv.0.weight", "inc.double_conv.1.bias", "inc.double_conv.1.weight", "inc.double_conv.3.weight", "inc.double_conv.4.bias", "inc.double_conv.4.weight", "outc.bias", "outc.weight", "sa1.ff_self.0.bias", "sa1.ff_self.0.weight", "sa1.ff_self.1.bias", "sa1.ff_self.1.weight", "sa1.ff_self.3.bias", "sa1.ff_self.3.weight", "sa1.ln.bias", "sa1.ln.weight", "sa1.mha.in_proj_bias", "sa1.mha.in_proj_weight", "sa1.mha.out_proj.bias", "sa1.mha.out_proj.weight", "sa2.ff_self.0.bias", "sa2.ff_self.0.weight", "sa2.ff_self.1.bias", "sa2.ff_self.1.weight", "sa2.ff_self.3.bias", "sa2.ff_self.3.weight", "sa2.ln.bias", "sa2.ln.weight", "sa2.mha.in_proj_bias", "sa2.mha.in_proj_weight", "sa2.mha.out_proj.bias", "sa2.mha.out_proj.weight", "sa3.ff_self.0.bias", "sa3.ff_self.0.weight", "sa3.ff_self.1.bias", "sa3.ff_self.1.weight", "sa3.ff_self.3.bias", "sa3.ff_self.3.weight", "sa3.ln.bias", "sa3.ln.weight", "sa3.mha.in_proj_bias", "sa3.mha.in_proj_weight", "sa3.mha.out_proj.bias", "sa3.mha.out_proj.weight", "sa4.ff_self.0.bias", "sa4.ff_self.0.weight", "sa4.ff_self.1.bias", "sa4.ff_self.1.weight", "sa4.ff_self.3.bias", "sa4.ff_self.3.weight", "sa4.ln.bias", "sa4.ln.weight", "sa4.mha.in_proj_bias", "sa4.mha.in_proj_weight", "sa4.mha.out_proj.bias", "sa4.mha.out_proj.weight", "sa5.ff_self.0.bias", "sa5.ff_self.0.weight", "sa5.ff_self.1.bias", "sa5.ff_self.1.weight", "sa5.ff_self.3.bias", "sa5.ff_self.3.weight", "sa5.ln.bias", "sa5.ln.weight", "sa5.mha.in_proj_bias", "sa5.mha.in_proj_weight", "sa5.mha.out_proj.bias", "sa5.mha.out_proj.weight", "sa6.ff_self.0.bias", "sa6.ff_self.0.weight", "sa6.ff_self.1.bias", "sa6.ff_self.1.weight", "sa6.ff_self.3.bias", "sa6.ff_self.3.weight", "sa6.ln.bias", "sa6.ln.weight", "sa6.mha.in_proj_bias", "sa6.mha.in_proj_weight", "sa6.mha.out_proj.bias", "sa6.mha.out_proj.weight", "up1.conv.0.double_conv.0.weight", "up1.conv.0.double_conv.1.bias", "up1.conv.0.double_conv.1.weight", "up1.conv.0.double_conv.3.weight", "up1.conv.0.double_conv.4.bias", "up1.conv.0.double_conv.4.weight", "up1.conv.1.double_conv.0.weight", "up1.conv.1.double_conv.1.bias", "up1.conv.1.double_conv.1.weight", "up1.conv.1.double_conv.3.weight", "up1.conv.1.double_conv.4.bias", "up1.conv.1.double_conv.4.weight", "up1.emb_layer.1.bias", "up1.emb_layer.1.weight", "up2.conv.0.double_conv.0.weight", "up2.conv.0.double_conv.1.bias", "up2.conv.0.double_conv.1.weight", "up2.conv.0.double_conv.3.weight", "up2.conv.0.double_conv.4.bias", "up2.conv.0.double_conv.4.weight", "up2.conv.1.double_conv.0.weight", "up2.conv.1.double_conv.1.bias", "up2.conv.1.double_conv.1.weight", "up2.conv.1.double_conv.3.weight", "up2.conv.1.double_conv.4.bias", "up2.conv.1.double_conv.4.weight", "up2.emb_layer.1.bias", "up2.emb_layer.1.weight", "up3.conv.0.double_conv.0.weight", "up3.conv.0.double_conv.1.bias", "up3.conv.0.double_conv.1.weight", "up3.conv.0.double_conv.3.weight", "up3.conv.0.double_conv.4.bias", "up3.conv.0.double_conv.4.weight", "up3.conv.1.double_conv.0.weight", "up3.conv.1.double_conv.1.bias", "up3.conv.1.double_conv.1.weight", "up3.conv.1.double_conv.3.weight", "up3.conv.1.double_conv.4.bias", "up3.conv.1.double_conv.4.weight", "up3.emb_layer.1.bias", "up3.emb_layer.1.weight"];

export async function upload_params(dirname: string) {
    const url = "https://web-diffusion-worker.0xtimmy.workers.dev/parameters";
    const model = "pokemon";
    const overwrite = false;
    await Promise.all(params.map(async (parameter): Promise<void> => {
        let write = true;
        if(!overwrite) {
            const check = await fetch(`${url}/${model}/${parameter}/shape`, { method: "GET" });
            if(check.status == 200) write = false;
        }
        if(write) {
            const data = await (await fetch(`${dirname}/${model}/${parameter}`)).json();
            const shape = torch.tensor(data).shape;
            const typedData = new Float32Array(data.flat(4));
            const data_res = await fetch(`${url}/${model}/${parameter}/data`, {
                method: "POST",
                body: typedData.buffer,
                headers: {
                    "Content-Type": "arrayBuffer",
                    "WRITE-PASSCODE": ""
                }
            });
            if(data_res.status == 200) console.log(`✅ Data response for parameter: ${parameter}`, await data_res.text());
            else {
                console.error(`🚩 Data response for parameter: ${parameter}`, await data_res.text());
                throw new Error(`Bad response`)
            }

            const shape_res = await fetch(`${url}/${model}/${parameter}/shape`, {
                method: "POST",
                body: JSON.stringify(shape),
                headers: {
                    "Content-Type": "json",
                    "WRITE-PASSCODE": ""
                }
            });
            if(data_res.status == 200) console.log(`✅ Shape response for parameter: ${parameter}`, await shape_res.text());
            else {
                console.error(`🚩 Data response for parameter: ${parameter}`, await data_res.text());
                throw new Error(`Bad response`);
            }
        }
        return;
    }));
    console.log("🏁 Successfully uploaded all parameters");
}

export async function get_params() {
    const url = "https://web-diffusion-worker.0xtimmy.workers.dev/parameters";
    const model = "pokemon";
    const parameter = params[0];
    const data = await fetch(`${url}/${model}/${parameter}/data`, {
        method: "GET",
    });
    const shape = await (await fetch(`${url}/${model}/${parameter}/shape`, {
        method: "GET",
    })).json();
    const buf = await data.arrayBuffer();
    const arr = new Float32Array(buf);
    const tensor = torch.tensor(Array.from(arr)).view(shape);
    console.log("out tensor: ", shape, arr)
}