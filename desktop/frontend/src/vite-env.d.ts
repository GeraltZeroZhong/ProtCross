/// <reference types="vite/client" />

declare module "molstar/build/viewer/molstar" {
  export const Viewer: {
    create(element: HTMLElement, options?: Record<string, unknown>): Promise<any>;
  };
}
