self.onmessage = async (e) => {
    console.log("Worker has recieved message: ", e);

    //await this.storage.mapReadAsync();
    //const data = this.storage.getTypedArray(this.dtype);
        
    /*
    console.log("loading weights...");
    model = new UNet();
    await model.loadStateDictFromURL("./parameters/pokemon");
    console.log("✅ done loading weights");

    diffuser = new Diffusion({ noise_steps: 10 });
    diffuser.sample(model, (img, step_num) => {
        console.log("sampled: ", step_num);
    })
    */
}