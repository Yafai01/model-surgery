// What the readers need from a file: its size and random-access byte ranges.
// Browser File objects satisfy this; so does RemoteFile (Hugging Face Hub range requests).

export interface ByteRange {
  arrayBuffer(): Promise<ArrayBuffer>;
}

export interface ByteSource {
  readonly size: number;
  /** Initial header read size; remote sources use a small value and grow on demand. */
  readonly headerChunk?: number;
  slice(start: number, end: number): ByteRange;
}

export interface FileLike extends ByteSource {
  readonly name: string;
  readonly webkitRelativePath?: string;
  arrayBuffer(): Promise<ArrayBuffer>;
}
