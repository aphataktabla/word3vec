// cooc_cc100.cpp
// Fast GloVe-style word co-occurrence counts from cc100-style txt.zst shards.
// Each line = one document. No JSON parsing.
// Symmetric counts, window=5, no distance weighting, sparse spill+merge.
//
// Build:
//   g++ -O3 -march=native -std=c++17 -pthread cooc_cc100.cpp -lzstd -o cooc_cc100
//
// Run:
//   ./cooc_cc100 --input_dir /mnt/nvme/bench/cc100_en_200k --work_dir /mnt/nvme/work/cc100_200k --out_dir /mnt/nvme/out/cc100_200k \
//     --threads 32 --vmax 300000 --window 5 --buckets 512 --flush_entries 3000000

#include <zstd.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <queue>

namespace fs = std::filesystem;

// ---------------- CLI ----------------
struct Args {
  std::string input_dir, work_dir, out_dir;
  int threads = 16;
  int window = 5;
  uint32_t vmax = 300000;
  uint32_t buckets = 512;
  uint64_t flush_entries = 3'000'000;
  bool skip_diag = true;
};

static void die(const std::string& msg) {
  std::cerr << "ERROR: " << msg << "\n";
  std::exit(1);
}

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) die(std::string("Missing value for ") + name);
      return std::string(argv[++i]);
    };
    if (k == "--input_dir") a.input_dir = need("--input_dir");
    else if (k == "--work_dir") a.work_dir = need("--work_dir");
    else if (k == "--out_dir") a.out_dir = need("--out_dir");
    else if (k == "--threads") a.threads = std::stoi(need("--threads"));
    else if (k == "--window") a.window = std::stoi(need("--window"));
    else if (k == "--vmax") a.vmax = (uint32_t)std::stoul(need("--vmax"));
    else if (k == "--buckets") a.buckets = (uint32_t)std::stoul(need("--buckets"));
    else if (k == "--flush_entries") a.flush_entries = (uint64_t)std::stoull(need("--flush_entries"));
    else if (k == "--skip_diag") {
      auto v = need("--skip_diag");
      a.skip_diag = (v == "1" || v == "true" || v == "True");
    } else if (k == "--help") {
      std::cout <<
        "Usage: ./cooc_cc100 --input_dir DIR --work_dir DIR --out_dir DIR [options]\n"
        "Options: --threads N --window W --vmax V --buckets B --flush_entries K --skip_diag true/false\n";
      std::exit(0);
    } else die("Unknown arg: " + k);
  }
  if (a.input_dir.empty() || a.work_dir.empty() || a.out_dir.empty())
    die("Must set --input_dir --work_dir --out_dir");
  if (a.window <= 0) die("--window must be > 0");
  if (a.threads <= 0) die("--threads must be > 0");
  if (a.buckets == 0) die("--buckets must be > 0");
  return a;
}

// ---------------- Zstd line reader ----------------
struct ZstdLineReader {
  FILE* fp = nullptr;
  ZSTD_DCtx* dctx = nullptr;
  std::vector<char> inBuf, outBuf;
  ZSTD_inBuffer input{nullptr,0,0};
  size_t outPos=0, outSize=0;
  std::string carry;

  explicit ZstdLineReader(const std::string& path, size_t inChunk=1<<20, size_t outChunk=1<<20) {
    fp = std::fopen(path.c_str(), "rb");
    if (!fp) die("Failed to open: " + path + " : " + std::strerror(errno));
    dctx = ZSTD_createDCtx();
    if (!dctx) die("ZSTD_createDCtx failed");
    inBuf.resize(inChunk);
    outBuf.resize(outChunk);
    input.src = inBuf.data();
    input.size = 0;
    input.pos = 0;
  }
  ~ZstdLineReader() {
    if (dctx) ZSTD_freeDCtx(dctx);
    if (fp) std::fclose(fp);
  }
  bool refillInput() {
    size_t n = std::fread(inBuf.data(), 1, inBuf.size(), fp);
    if (n == 0) return false;
    input.src = inBuf.data();
    input.size = n;
    input.pos = 0;
    return true;
  }
  bool decompressMore() {
    if (input.pos >= input.size) {
      if (!refillInput()) return false;
    }
    ZSTD_outBuffer output{outBuf.data(), outBuf.size(), 0};
    size_t rc = ZSTD_decompressStream(dctx, &output, &input);
    if (ZSTD_isError(rc)) die(std::string("ZSTD_decompressStream: ") + ZSTD_getErrorName(rc));
    outSize = output.pos;
    outPos = 0;
    return outSize > 0;
  }
  bool nextLine(std::string& line) {
    while (true) {
      auto p = carry.find('\n');
      if (p != std::string::npos) {
        line = carry.substr(0, p);
        carry.erase(0, p+1);
        return true;
      }
      if (outPos >= outSize) {
        if (!decompressMore()) {
          if (!carry.empty()) {
            line.swap(carry);
            carry.clear();
            return true;
          }
          return false;
        }
      }
      size_t n = std::min((size_t)(outSize - outPos), (size_t)(1<<20));
      carry.append(outBuf.data() + outPos, n);
      outPos += n;
    }
  }
};

// ---------------- Tokenizer ----------------
// Simple ASCII-ish tokenizer: lowercases, keeps [a-z0-9'].
static inline bool is_tok_char(unsigned char c) {
  return (c>='a'&&c<='z') || (c>='0'&&c<='9') || (c=='\'');
}
static std::vector<std::string> tokenize_words(const std::string& text) {
  std::vector<std::string> out;
  out.reserve(256);
  std::string cur; cur.reserve(32);
  for (unsigned char uc : text) {
    unsigned char c = uc;
    if (c>='A'&&c<='Z') c = (unsigned char)(c - 'A' + 'a');
    if (is_tok_char(c)) cur.push_back((char)c);
    else {
      if (!cur.empty()) { out.push_back(std::move(cur)); cur.clear(); cur.reserve(32); }
    }
  }
  if (!cur.empty()) out.push_back(std::move(cur));
  return out;
}

// ---------------- Utils ----------------
static std::vector<std::string> list_zst_files(const std::string& root) {
  std::vector<std::string> files;
  for (auto& p : fs::recursive_directory_iterator(root)) {
    if (!p.is_regular_file()) continue;
    auto path = p.path().string();
    if (path.size() >= 8 && path.substr(path.size()-8) == ".txt.zst")
      files.push_back(path);
  }
  std::sort(files.begin(), files.end());
  return files;
}

static inline uint64_t pack_key(uint32_t i, uint32_t j) {
  return (uint64_t(i) << 32) | uint64_t(j);
}
static inline uint64_t mix64(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33; return x;
}

// ---------------- Pass 1: vocab ----------------
struct WordCount { std::string w; uint64_t c; };

static void build_vocab_topk(const Args& args, const std::vector<std::string>& files) {
  std::cout << "[Pass1] Counting words...\n";
  std::unordered_map<std::string, uint64_t> counts;
  counts.reserve(5'000'000);

  uint64_t docs = 0;
  for (const auto& f : files) {
    ZstdLineReader r(f);
    std::string line;
    while (r.nextLine(line)) {
      docs++;
      if ((docs % 500'000) == 0) std::cerr << "\r[Pass1] docs: " << docs << std::flush;
      auto toks = tokenize_words(line);
      for (auto& w : toks) counts[w] += 1;
    }
  }
  std::cerr << "\n[Pass1] Unique words: " << counts.size() << "\n";

  std::vector<WordCount> v; v.reserve(counts.size());
  for (auto& kv : counts) v.push_back({kv.first, kv.second});

  if (v.size() > args.vmax) {
    std::nth_element(v.begin(), v.begin() + args.vmax, v.end(),
      [](const WordCount& a, const WordCount& b){ return a.c > b.c; });
    v.resize(args.vmax);
  }
  std::sort(v.begin(), v.end(), [](const WordCount& a, const WordCount& b){
    if (a.c != b.c) return a.c > b.c;
    return a.w < b.w;
  });

  fs::create_directories(args.out_dir);
  std::string vocab_path = (fs::path(args.out_dir) / "vocab.tsv").string();
  std::ofstream out(vocab_path);
  if (!out) die("Failed to write " + vocab_path);

  for (uint32_t id = 0; id < (uint32_t)v.size(); id++) {
    out << v[id].w << "\t" << id << "\t" << v[id].c << "\n";
  }
  out.close();
  std::cout << "[Pass1] Wrote vocab: " << vocab_path << " (V=" << v.size() << ")\n";
}

static std::unordered_map<std::string, uint32_t> load_vocab_map(const Args& args) {
  std::string vocab_path = (fs::path(args.out_dir) / "vocab.tsv").string();
  std::ifstream in(vocab_path);
  if (!in) die("Missing vocab.tsv; run pass1 first.");
  std::unordered_map<std::string, uint32_t> map;
  map.reserve(args.vmax * 1.3);

  std::string word; uint32_t id; uint64_t c;
  while (in >> word >> id >> c) map.emplace(word, id);
  std::cout << "[Info] Loaded vocab map size=" << map.size() << "\n";
  return map;
}

// ---------------- PairMap: uint64 -> uint32 ----------------
struct PairMap {
  std::vector<uint64_t> keys;
  std::vector<uint32_t> vals;
  uint64_t mask = 0, used = 0;
  static constexpr uint64_t EMPTY = 0xffffffffffffffffULL;

  explicit PairMap(uint64_t cap_pow2 = 1<<20) { init(cap_pow2); }

  void init(uint64_t cap_pow2) {
    uint64_t cap = 1; while (cap < cap_pow2) cap <<= 1;
    keys.assign(cap, EMPTY);
    vals.assign(cap, 0);
    mask = cap - 1;
    used = 0;
  }
  void maybe_grow() { if (used * 10 < (uint64_t)keys.size() * 7) return; rehash(keys.size()*2); }
  void rehash(uint64_t new_cap) {
    std::vector<uint64_t> ok = std::move(keys);
    std::vector<uint32_t> ov = std::move(vals);
    init(new_cap);
    for (size_t i = 0; i < ok.size(); i++) if (ok[i] != EMPTY) add(ok[i], ov[i]);
  }
  void add(uint64_t key, uint32_t inc=1) {
    maybe_grow();
    uint64_t pos = mix64(key) & mask;
    while (true) {
      if (keys[pos] == EMPTY) { keys[pos]=key; vals[pos]=inc; used++; return; }
      if (keys[pos] == key) { vals[pos]+=inc; return; }
      pos = (pos + 1) & mask;
    }
  }
  size_t size() const { return (size_t)used; }
  void to_vector(std::vector<std::pair<uint64_t,uint32_t>>& out) {
    out.clear(); out.reserve(used);
    for (size_t i = 0; i < keys.size(); i++) if (keys[i] != EMPTY) out.emplace_back(keys[i], vals[i]);
  }
  void clear() {
    std::fill(keys.begin(), keys.end(), EMPTY);
    std::fill(vals.begin(), vals.end(), 0);
    used = 0;
  }
};

struct RunWriter {
  std::string runs_dir;
  uint32_t buckets, tid, run_id = 0;

  RunWriter(std::string dir, uint32_t b, uint32_t t) : runs_dir(std::move(dir)), buckets(b), tid(t) {}

  void write_runs(const std::vector<std::pair<uint64_t,uint32_t>>& items) {
    std::vector<std::vector<std::pair<uint64_t,uint32_t>>> parts(buckets);
    for (auto& kv : items) {
      uint32_t b = (uint32_t)(mix64(kv.first) & (buckets - 1));
      parts[b].push_back(kv);
    }
    fs::create_directories(runs_dir);
    for (uint32_t b = 0; b < buckets; b++) {
      if (parts[b].empty()) continue;
      auto& vec = parts[b];
      std::sort(vec.begin(), vec.end(), [](auto& a, auto& c){ return a.first < c.first; });

      char name[256];
      std::snprintf(name, sizeof(name), "t%03u_b%04u_r%06u.bin", tid, b, run_id);
      std::string path = (fs::path(runs_dir) / name).string();

      std::ofstream out(path, std::ios::binary);
      if (!out) die("Failed to write run: " + path);
      for (auto& rec : vec) {
        out.write((const char*)&rec.first, sizeof(uint64_t));
        out.write((const char*)&rec.second, sizeof(uint32_t));
      }
    }
    run_id++;
  }
};

struct WorkQueue {
  std::vector<std::string> files;
  std::atomic<size_t> idx{0};
  explicit WorkQueue(std::vector<std::string> f) : files(std::move(f)) {}
  bool next(std::string& out) {
    size_t i = idx.fetch_add(1);
    if (i >= files.size()) return false;
    out = files[i];
    return true;
  }
};

static void pass2_worker(const Args& args,
                         const std::unordered_map<std::string,uint32_t>& vocab,
                         WorkQueue& q, uint32_t tid,
                         std::atomic<uint64_t>& total_docs) {
  PairMap map(1<<22);
  std::vector<std::pair<uint64_t,uint32_t>> dump;
  RunWriter writer((fs::path(args.work_dir) / "runs").string(), args.buckets, tid);

  std::string file;
  while (q.next(file)) {
    ZstdLineReader r(file);
    std::string doc;
    while (r.nextLine(doc)) {
      uint64_t d = total_docs.fetch_add(1) + 1;
      if ((d % 500000) == 0 && tid == 0) std::cerr << "\r[Pass2] docs: " << d << std::flush;

      auto toks = tokenize_words(doc);

      std::vector<uint32_t> ids;
      ids.reserve(toks.size());
      for (auto& w : toks) {
        auto it = vocab.find(w);
        if (it != vocab.end()) ids.push_back(it->second);
      }
      if (ids.size() < 2) continue;

      const int W = args.window;
      for (size_t t = 0; t < ids.size(); t++) {
        uint32_t i = ids[t];
        size_t right = std::min(ids.size(), t + (size_t)W + 1);
        for (size_t u = t + 1; u < right; u++) {
          uint32_t j = ids[u];
          if (args.skip_diag && i == j) continue;
          map.add(pack_key(i, j), 1);
          map.add(pack_key(j, i), 1);
        }
        if (map.size() >= args.flush_entries) {
          map.to_vector(dump);
          writer.write_runs(dump);
          map.clear();
        }
      }
    }
  }
  if (map.size() > 0) {
    map.to_vector(dump);
    writer.write_runs(dump);
    map.clear();
  }
}

// ---------------- Merge per bucket ----------------
struct RunReader {
  std::ifstream in;
  uint64_t key=0; uint32_t val=0; bool ok=false;
  explicit RunReader(const std::string& path) : in(path, std::ios::binary) {
    if (!in) die("Failed to open run: " + path);
    ok = read();
  }
  bool read() {
    if (!in.read((char*)&key, sizeof(uint64_t))) return false;
    if (!in.read((char*)&val, sizeof(uint32_t))) return false;
    return true;
  }
};
struct HeapItem { uint64_t key; uint32_t val; size_t ridx; bool operator>(const HeapItem& o) const { return key > o.key; } };

static std::vector<std::string> list_bucket_runs(const std::string& runs_dir, uint32_t bucket) {
  std::vector<std::string> out;
  char needle[16]; std::snprintf(needle, sizeof(needle), "_b%04u_", bucket);
  for (auto& p : fs::directory_iterator(runs_dir)) {
    if (!p.is_regular_file()) continue;
    std::string name = p.path().filename().string();
    if (name.find(needle) != std::string::npos && name.size() >= 4 && name.substr(name.size()-4)==".bin")
      out.push_back(p.path().string());
  }
  std::sort(out.begin(), out.end());
  return out;
}

static void merge_bucket(const Args& args, uint32_t bucket) {
  std::string runs_dir = (fs::path(args.work_dir) / "runs").string();
  auto run_files = list_bucket_runs(runs_dir, bucket);
  if (run_files.empty()) return;

  std::vector<std::unique_ptr<RunReader>> readers;
  readers.reserve(run_files.size());
  for (auto& f : run_files) readers.emplace_back(std::make_unique<RunReader>(f));

  std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;
  for (size_t i = 0; i < readers.size(); i++) if (readers[i]->ok) heap.push({readers[i]->key, readers[i]->val, i});

  fs::create_directories(args.out_dir);
  fs::create_directories(fs::path(args.out_dir) / "cooc");
  char outname[64]; std::snprintf(outname, sizeof(outname), "b%04u.bin", bucket);
  std::string outpath = (fs::path(args.out_dir) / "cooc" / outname).string();

  std::ofstream out(outpath, std::ios::binary);
  if (!out) die("Failed to write: " + outpath);

  uint64_t cur_key=0, cur_sum=0; bool has=false;
  while (!heap.empty()) {
    auto it = heap.top(); heap.pop();

    if (!has) { cur_key = it.key; cur_sum = it.val; has=true; }
    else if (it.key == cur_key) cur_sum += it.val;
    else {
      uint32_t i = (uint32_t)(cur_key >> 32);
      uint32_t j = (uint32_t)(cur_key & 0xffffffffu);
      out.write((const char*)&i, sizeof(uint32_t));
      out.write((const char*)&j, sizeof(uint32_t));
      out.write((const char*)&cur_sum, sizeof(uint64_t));
      cur_key = it.key; cur_sum = it.val;
    }

    auto& rr = readers[it.ridx];
    rr->ok = rr->read();
    if (rr->ok) heap.push({rr->key, rr->val, it.ridx});
  }

  if (has) {
    uint32_t i = (uint32_t)(cur_key >> 32);
    uint32_t j = (uint32_t)(cur_key & 0xffffffffu);
    out.write((const char*)&i, sizeof(uint32_t));
    out.write((const char*)&j, sizeof(uint32_t));
    out.write((const char*)&cur_sum, sizeof(uint64_t));
  }
}

// ---------------- Main ----------------
int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  if ((args.buckets & (args.buckets - 1)) != 0) {
    std::cerr << "[Warn] --buckets not power of two; power-of-two recommended.\n";
  }

  auto files = list_zst_files(args.input_dir);
  if (files.empty()) die("No .txt.zst files found under " + args.input_dir);
  std::cout << "[Info] Found " << files.size() << " input files\n";

  fs::create_directories(args.work_dir);
  fs::create_directories(args.out_dir);

  std::string vocab_path = (fs::path(args.out_dir) / "vocab.tsv").string();
  if (!fs::exists(vocab_path)) build_vocab_topk(args, files);
  else std::cout << "[Info] vocab.tsv exists; skipping pass1\n";

  auto vocab = load_vocab_map(args);

  std::cout << "[Pass2] Counting co-occurrences (symmetric)...\n";
  WorkQueue q(files);
  std::atomic<uint64_t> total_docs{0};
  std::vector<std::thread> th;
  th.reserve(args.threads);
  for (int t = 0; t < args.threads; t++)
    th.emplace_back(pass2_worker, std::cref(args), std::cref(vocab), std::ref(q), (uint32_t)t, std::ref(total_docs));
  for (auto& t : th) t.join();
  std::cerr << "\n[Pass2] Done docs: " << total_docs.load() << "\n";

  std::cout << "[Merge] Merging buckets...\n";
  std::atomic<uint32_t> bidx{0};
  auto merge_worker = [&]() {
    while (true) {
      uint32_t b = bidx.fetch_add(1);
      if (b >= args.buckets) break;
      merge_bucket(args, b);
      if ((b % 64) == 0) std::cerr << "\r[Merge] bucket " << b << "/" << args.buckets << std::flush;
    }
  };

  int merge_threads = std::min(args.threads, 16);
  std::vector<std::thread> mt;
  for (int i = 0; i < merge_threads; i++) mt.emplace_back(merge_worker);
  for (auto& t : mt) t.join();
  std::cerr << "\n[Merge] Done.\n";

  std::cout << "[Done] Outputs:\n"
            << "  " << vocab_path << "\n"
            << "  " << (fs::path(args.out_dir) / "cooc").string() << "\n";
  return 0;
}
