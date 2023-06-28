import * as torch from "@/torch";

export async function init_device() {
    if (!await torch.initWebGPUAsync()) {
        console.warn("‼️ WebGPU is not supported.");
        return false;
    }
    console.log("✅ WebGPU initialized");
    return true;
}