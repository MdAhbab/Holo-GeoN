import type {
  DdgResponse,
  ScanResponse,
  EpistasisResponse,
  IdrResponse,
  CompareResponse,
} from './types'

/**
 * Module-level store for the latest computed results.
 * Components read/write these directly; the Export tab checks them.
 * We use a simple pub/sub to allow components to subscribe to changes.
 */

type Listener = () => void

export interface Store {
  ddg: DdgResponse | null
  scan: (ScanResponse & { sequence?: string }) | null
  epistasis: (EpistasisResponse & { wt_sequence?: string }) | null
  idr: IdrResponse | null
  compare: CompareResponse | null
}

const store: Store = {
  ddg: null,
  scan: null,
  epistasis: null,
  idr: null,
  compare: null,
}

const listeners = new Set<Listener>()

function notify() {
  listeners.forEach((fn) => fn())
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

export function getStore(): Readonly<Store> {
  return store
}

export function getDdg(): DdgResponse | null {
  return store.ddg
}

export function setDdg(result: DdgResponse) {
  store.ddg = result
  notify()
}

export function getScan(): (ScanResponse & { sequence?: string }) | null {
  return store.scan
}

export function setScan(result: ScanResponse & { sequence?: string }) {
  store.scan = result
  notify()
}

export function getEpistasis(): (EpistasisResponse & { wt_sequence?: string }) | null {
  return store.epistasis
}

export function setEpistasis(result: EpistasisResponse & { wt_sequence?: string }) {
  store.epistasis = result
  notify()
}

export function getIdr(): IdrResponse | null {
  return store.idr
}

export function setIdr(result: IdrResponse) {
  store.idr = result
  notify()
}

export function getCompare(): CompareResponse | null {
  return store.compare
}

export function setCompare(result: CompareResponse) {
  store.compare = result
  notify()
}
