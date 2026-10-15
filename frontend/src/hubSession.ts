// The optional Hugging Face access token lives only in this tab's memory: never stored, never logged,
// and the engine sends it only to huggingface.co.
let token: string | null = null;

export const getHubToken = () => token;
export const setHubToken = (t: string | null) => {
  token = t && t.trim() ? t.trim() : null;
};
