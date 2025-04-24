export default function handler(req, res) {
  const { storage, cpu, bandwidth } = req.query;

  // Convert the parameters to numbers (defaulting to 0 if absent)
  const storageGB = parseFloat(storage) || 0;
  const cpuCores = parseFloat(cpu) || 0;
  const bandwidthTB = parseFloat(bandwidth) || 0;

  // Calculate the cost:
  // - Storage: $0.02 per GB
  // - CPU: $5 per core
  // - Bandwidth: $10 per TB
  const cost = (storageGB * 0.02) + (cpuCores * 5) + (bandwidthTB * 10);

  res.status(200).json({ cost: cost.toFixed(2) });
}