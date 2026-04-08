import rawCSV from '../demo_data/winner_loser_sampled_exchange0_55HITs.csv?raw'

// Eagerly import all local cow videos – Vite resolves these at build time
const cowVideoFilesMP4 = import.meta.glob<string>(
  '../demo_data/compressed_cow/*.MP4',
  { eager: true, query: '?url', import: 'default' }
)
const cowVideoFilesmp4 = import.meta.glob<string>(
  '../demo_data/compressed_cow/*.mp4',
  { eager: true, query: '?url', import: 'default' }
)
const cowVideoFiles: Record<string, string> = { ...cowVideoFilesMP4, ...cowVideoFilesmp4 }

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ConsensusData {
  pairKey: string
  minCow: string
  maxCow: string
  /** Canonical signed scores: +degree means maxCow more lame, -degree means minCow more lame */
  scores: number[]
  mean: number
  median: number
  stdev: number
  /** % of responses whose direction matches the majority */
  agreePercent: number
  count: number
}

export interface FeedbackResult {
  message: string
  emoji: string
  type: 'great' | 'good' | 'interesting' | 'unique'
  details: string
  /** Which side the consensus favors (relative to left/right display) */
  consensusFavor: 'left' | 'right' | 'tied'
  consensusMean: number
  userCanonical: number
  agreePercent: number
}

export interface DemoConsensusPair {
  cow_L: string
  cow_R: string
  cow_L_URL: string
  cow_R_URL: string
  pairKey: string
  consensusData: ConsensusData
}

// ---------------------------------------------------------------------------
// Build cow-ID → video URL map
// ---------------------------------------------------------------------------

function buildCowVideoMap(): Map<string, string> {
  const map = new Map<string, string>()
  for (const [filePath, url] of Object.entries(cowVideoFiles)) {
    // filename pattern: compressed_..._<clip>_<cowId>.<ext>
    const match = filePath.match(/_(\d{4})\.[Mm][Pp]4$/)
    if (match) {
      const cowId = match[1]
      if (!map.has(cowId)) {
        map.set(cowId, url)
      }
    }
  }
  return map
}

// ---------------------------------------------------------------------------
// Parse CSV and compute consensus per canonical pair
// ---------------------------------------------------------------------------

function parseConsensus(): Map<string, ConsensusData> {
  const lines = rawCSV.trim().split('\n')
  const pairScores = new Map<string, number[]>()

  for (const line of lines.slice(1)) {
    const parts = line.trim().split(',')
    if (parts.length < 3) continue

    const winner = parts[0].trim()
    const loser = parts[1].trim()
    const degree = parseInt(parts[2].trim(), 10)
    if (isNaN(degree) || !winner || !loser) continue

    const wNum = parseInt(winner, 10)
    const lNum = parseInt(loser, 10)
    const minNum = Math.min(wNum, lNum)
    const maxNum = Math.max(wNum, lNum)
    const minCow = String(minNum)
    const maxCow = String(maxNum)
    const key = `${minCow}_${maxCow}`

    // Canonical score convention:
    //   positive → maxCow is more lame (minCow wins/is healthier)
    //   negative → minCow is more lame (maxCow wins/is healthier)
    const canonicalScore = winner === minCow ? +degree : -degree

    const existing = pairScores.get(key)
    if (existing) {
      existing.push(canonicalScore)
    } else {
      pairScores.set(key, [canonicalScore])
    }
  }

  const consensus = new Map<string, ConsensusData>()

  for (const [key, scores] of pairScores.entries()) {
    const [minCow, maxCow] = key.split('_')
    const n = scores.length
    const mean = scores.reduce((a, b) => a + b, 0) / n

    const sorted = [...scores].sort((a, b) => a - b)
    const mid = Math.floor(sorted.length / 2)
    const median =
      sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid]

    const stdev = Math.sqrt(
      scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / n
    )

    // Agreement: what fraction match the majority direction?
    const majoritySign = Math.sign(mean)
    const agreedCount =
      majoritySign === 0
        ? scores.filter(s => s === 0).length
        : scores.filter(s => Math.sign(s) === majoritySign).length
    const agreePercent = (agreedCount / n) * 100

    consensus.set(key, {
      pairKey: key,
      minCow,
      maxCow,
      scores,
      mean,
      median,
      stdev,
      agreePercent,
      count: n,
    })
  }

  return consensus
}

// ---------------------------------------------------------------------------
// Cow ranking
// ---------------------------------------------------------------------------

export interface CowRanking {
  cowId: string
  /** 1 = most lame */
  rank: number
  /** Average lameness signal: positive = more lame, range roughly -3…+3 */
  rawScore: number
  /** 0–1 normalized across all cows; 1 = most lame */
  normalizedScore: number
  severity: 'healthy' | 'mild' | 'moderate' | 'severe'
  /** Total individual judgments this cow participated in */
  comparisons: number
  /** Times judged as the less-lame / healthier cow */
  wins: number
  /** Times judged as the more-lame cow */
  losses: number
  /** Times judged equal (degree 0) */
  ties: number
  videoUrl?: string
}

/**
 * Derive a global lameness ranking for every cow that has real comparison data.
 *
 * For each comparison in the CSV, the loser (more lame) receives a positive
 * lameness signal equal to the degree (1–3), and the winner (less lame)
 * receives a negative signal.  The average signal per cow is then
 * min-max normalised to 0–1 and bucketed into severity tiers.
 */
export function getCowRankings(): CowRanking[] {
  const consensus = getConsensusData()
  const videoMap = getCowVideoMap()

  // Accumulate signed lameness signals per cow
  // canonical score for pair (minCow, maxCow):
  //   positive → maxCow is more lame
  //   negative → minCow is more lame
  const signals = new Map<string, number[]>()

  for (const pair of consensus.values()) {
    const { minCow, maxCow, scores } = pair
    for (const c of scores) {
      if (!signals.has(maxCow)) signals.set(maxCow, [])
      if (!signals.has(minCow)) signals.set(minCow, [])
      signals.get(maxCow)!.push(c)   // positive → maxCow more lame
      signals.get(minCow)!.push(-c)  // flipped  → minCow more lame when negative
    }
  }

  // Compute per-cow raw scores
  const rows: {
    cowId: string; raw: number; comparisons: number
    wins: number; losses: number; ties: number
  }[] = []

  for (const [cowId, sigs] of signals.entries()) {
    const raw = sigs.reduce((a, b) => a + b, 0) / sigs.length
    rows.push({
      cowId,
      raw,
      comparisons: sigs.length,
      wins:   sigs.filter(s => s < 0).length,   // less lame = win
      losses: sigs.filter(s => s > 0).length,   // more lame = loss
      ties:   sigs.filter(s => s === 0).length,
    })
  }

  // Sort descending by raw score (most lame first)
  rows.sort((a, b) => b.raw - a.raw)

  const minRaw = rows[rows.length - 1]?.raw ?? 0
  const maxRaw = rows[0]?.raw ?? 1
  const range = maxRaw - minRaw || 1

  return rows.map((item, idx) => {
    const n = (item.raw - minRaw) / range  // 0–1, 1 = most lame

    let severity: CowRanking['severity']
    if (n >= 0.75) severity = 'severe'
    else if (n >= 0.50) severity = 'moderate'
    else if (n >= 0.25) severity = 'mild'
    else severity = 'healthy'

    return {
      cowId: item.cowId,
      rank: idx + 1,
      rawScore: item.raw,
      normalizedScore: n,
      severity,
      comparisons: item.comparisons,
      wins: item.wins,
      losses: item.losses,
      ties: item.ties,
      videoUrl: videoMap.get(item.cowId),
    }
  })
}

// ---------------------------------------------------------------------------
// Module-level caches (initialised lazily)
// ---------------------------------------------------------------------------

let _cowVideoMap: Map<string, string> | null = null
let _consensusData: Map<string, ConsensusData> | null = null
let _demoPairs: DemoConsensusPair[] | null = null

export function getCowVideoMap(): Map<string, string> {
  if (!_cowVideoMap) _cowVideoMap = buildCowVideoMap()
  return _cowVideoMap
}

export function getConsensusData(): Map<string, ConsensusData> {
  if (!_consensusData) _consensusData = parseConsensus()
  return _consensusData
}

/**
 * Returns all valid demo pairs: both cows have a local video AND real human
 * comparison data exists in the CSV.  Pairs are returned in a stable order.
 */
export function getValidDemoPairs(): DemoConsensusPair[] {
  if (_demoPairs) return _demoPairs

  const videoMap = getCowVideoMap()
  const consensus = getConsensusData()

  _demoPairs = []
  for (const [key, data] of consensus.entries()) {
    const cow_L_URL = videoMap.get(data.minCow)
    const cow_R_URL = videoMap.get(data.maxCow)
    if (!cow_L_URL || !cow_R_URL) continue

    _demoPairs.push({
      cow_L: data.minCow,
      cow_R: data.maxCow,
      cow_L_URL,
      cow_R_URL,
      pairKey: key,
      consensusData: data,
    })
  }

  return _demoPairs
}

// ---------------------------------------------------------------------------
// Feedback computation
// ---------------------------------------------------------------------------

/**
 * Compare a user's submitted answer against the crowd consensus for a pair.
 *
 * @param leftCowId  - The cow ID shown on the LEFT side
 * @param rightCowId - The cow ID shown on the RIGHT side
 * @param userValue  - The value on the 7-point scale:
 *                     negative (-1 to -3) = LEFT cow more lame
 *                     0 = equal / cannot decide
 *                     positive (1 to 3)   = RIGHT cow more lame
 */
export function computeFeedback(
  leftCowId: string,
  rightCowId: string,
  userValue: number
): FeedbackResult | null {
  const consensus = getConsensusData()

  const lNum = parseInt(leftCowId, 10)
  const rNum = parseInt(rightCowId, 10)
  const minCow = String(Math.min(lNum, rNum))
  const maxCow = String(Math.max(lNum, rNum))
  const key = `${minCow}_${maxCow}`

  const data = consensus.get(key)
  if (!data) return null

  // Convert userValue → canonical score
  // canonical positive = maxCow is more lame
  // if leftCow is minCow: user positive (right more lame = maxCow) → same sign
  // if leftCow is maxCow: user positive (right more lame = minCow) → flip sign
  const userCanonical = leftCowId === minCow ? userValue : -userValue

  const distance = Math.abs(userCanonical - data.mean)
  const directionMatch =
    userCanonical === 0 ||
    data.mean === 0 ||
    Math.sign(userCanonical) === Math.sign(data.mean)

  // Which side does the consensus say is more lame? (relative to display order)
  let consensusFavor: FeedbackResult['consensusFavor']
  if (Math.abs(data.mean) < 0.3) {
    consensusFavor = 'tied'
  } else if (leftCowId === minCow) {
    consensusFavor = data.mean > 0 ? 'right' : 'left'
  } else {
    consensusFavor = data.mean > 0 ? 'left' : 'right'
  }

  // Determine feedback tier
  let type: FeedbackResult['type']
  let message: string
  let emoji: string

  if (directionMatch && distance <= 1.0) {
    type = 'great'
    emoji = '🎯'
    message = "You're very close to what everyone else believes!"
  } else if (directionMatch && distance <= 2.0) {
    type = 'good'
    emoji = '👍'
    message = 'Your take aligns with the majority on this pair!'
  } else if (directionMatch && distance <= 3.5) {
    type = 'interesting'
    emoji = '🤔'
    message = 'You agree on which cow is lamer, but see the degree a bit differently.'
  } else if (!directionMatch && distance <= 2.5) {
    type = 'interesting'
    emoji = '🤔'
    message = 'Interesting perspective! Many raters actually saw this the other way.'
  } else {
    type = 'unique'
    emoji = '😮'
    message = 'Oops... this is quite a unique point of view!'
  }

  // Build human-readable details
  const agreePct = Math.round(data.agreePercent)
  const avgStrength = Math.abs(data.mean).toFixed(1)
  const favorStr =
    consensusFavor === 'tied'
      ? 'Raters were split – no clear consensus direction.'
      : `${agreePct}% of raters thought the ${consensusFavor} cow was more lame (avg. strength ${avgStrength}/3).`

  const details = favorStr

  return {
    message,
    emoji,
    type,
    details,
    consensusFavor,
    consensusMean: data.mean,
    userCanonical,
    agreePercent: data.agreePercent,
  }
}
