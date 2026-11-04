// The worker `createDefaultBvhWorker` starts: it builds one hierarchy per message, off the page's thread.
import { respondToBvhRequest, type BvhRequest, type BvhResponse } from "./bvhBuilder.js";

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<BvhRequest>) => void) | null;
  postMessage(message: BvhResponse, transfer: Transferable[]): void;
};

scope.onmessage = event => {
  const { response, transfer } = respondToBvhRequest(event.data);

  scope.postMessage(response, transfer);
};
