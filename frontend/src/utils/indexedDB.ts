// IndexedDB utility for storing mocap frame data
const DB_NAME = 'MocapStudioDB';
const DB_VERSION = 1;
const STORE_NAME = 'frameCache';

interface PersonData {
  vertices: number[][];
  cam_t?: number[];
  joints?: number[][];
}

interface CachedFrame {
  frame_idx: number;
  persons: Record<string, PersonData>;
  timestamp: number;
}

class MocapDB {
  private db: IDBDatabase | null = null;
  private initPromise: Promise<IDBDatabase> | null = null;

  async init(): Promise<IDBDatabase> {
    if (this.db) {
      return this.db;
    }

    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => {
        console.error('IndexedDB failed to open:', request.error);
        reject(request.error);
      };

      request.onsuccess = () => {
        this.db = request.result;
        console.log('IndexedDB opened successfully');
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Create object store if it doesn't exist
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const objectStore = db.createObjectStore(STORE_NAME, { keyPath: 'frame_idx' });
          objectStore.createIndex('timestamp', 'timestamp', { unique: false });
          console.log('IndexedDB object store created');
        }
      };
    });

    return this.initPromise;
  }

  async addFrame(frame: CachedFrame): Promise<void> {
    const db = await this.init();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const objectStore = transaction.objectStore(STORE_NAME);
      const request = objectStore.put(frame);

      request.onsuccess = () => resolve();
      request.onerror = () => {
        console.error('Failed to add frame to IndexedDB:', request.error);
        reject(request.error);
      };
    });
  }

  async getFrame(frame_idx: number): Promise<CachedFrame | undefined> {
    const db = await this.init();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const objectStore = transaction.objectStore(STORE_NAME);
      const request = objectStore.get(frame_idx);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => {
        console.error('Failed to get frame from IndexedDB:', request.error);
        reject(request.error);
      };
    });
  }

  async getAllFrames(): Promise<CachedFrame[]> {
    const db = await this.init();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const objectStore = transaction.objectStore(STORE_NAME);
      const request = objectStore.getAll();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => {
        console.error('Failed to get all frames from IndexedDB:', request.error);
        reject(request.error);
      };
    });
  }

  async getAllFrameIndices(): Promise<number[]> {
    const db = await this.init();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const objectStore = transaction.objectStore(STORE_NAME);
      const request = objectStore.getAllKeys();

      request.onsuccess = () => resolve(request.result as number[]);
      request.onerror = () => {
        console.error('Failed to get frame indices from IndexedDB:', request.error);
        reject(request.error);
      };
    });
  }

  async getFrameCount(): Promise<number> {
    const db = await this.init();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const objectStore = transaction.objectStore(STORE_NAME);
      const request = objectStore.count();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => {
        console.error('Failed to count frames in IndexedDB:', request.error);
        reject(request.error);
      };
    });
  }

  async clearAll(): Promise<void> {
    const db = await this.init();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const objectStore = transaction.objectStore(STORE_NAME);
      const request = objectStore.clear();

      request.onsuccess = () => {
        console.log('IndexedDB cleared');
        resolve();
      };
      request.onerror = () => {
        console.error('Failed to clear IndexedDB:', request.error);
        reject(request.error);
      };
    });
  }

  async deleteFrame(frame_idx: number): Promise<void> {
    const db = await this.init();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const objectStore = transaction.objectStore(STORE_NAME);
      const request = objectStore.delete(frame_idx);

      request.onsuccess = () => resolve();
      request.onerror = () => {
        console.error('Failed to delete frame from IndexedDB:', request.error);
        reject(request.error);
      };
    });
  }

  async estimateStorageUsage(): Promise<{ usage: number; quota: number; percentage: number }> {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      const usage = estimate.usage || 0;
      const quota = estimate.quota || 0;
      const percentage = quota > 0 ? (usage / quota) * 100 : 0;

      return { usage, quota, percentage };
    }

    return { usage: 0, quota: 0, percentage: 0 };
  }

  async requestPersistentStorage(): Promise<boolean> {
    if ('storage' in navigator && 'persist' in navigator.storage) {
      try {
        const isPersisted = await navigator.storage.persist();
        if (isPersisted) {
          console.log('Persistent storage granted');
        } else {
          console.log('Persistent storage denied');
        }
        return isPersisted;
      } catch (err) {
        console.error('Failed to request persistent storage:', err);
        return false;
      }
    }
    return false;
  }
}

// Export singleton instance
export const mocapDB = new MocapDB();
