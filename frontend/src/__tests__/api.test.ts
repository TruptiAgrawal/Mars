import { fetchCases, fetchVolume, runCase } from "../api";

describe("api client", () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  it("fetchCases calls /cases and returns parsed JSON", async () => {
    const mockCases = [{ id: "case_001", label: "Clean anterior lesion" }];
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockCases,
    });

    const result = await fetchCases();

    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining("/cases"));
    expect(result).toEqual(mockCases);
  });

  it("fetchVolume calls /cases/{id}/volume", async () => {
    const mockVolume = { shape: [2, 2, 2], slices: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] };
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockVolume,
    });

    const result = await fetchVolume("case_001");

    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining("/cases/case_001/volume"));
    expect(result).toEqual(mockVolume);
  });

  it("runCase calls /cases/{id}/run with tau query param", async () => {
    const mockResult = { case_id: "case_001", tau: 0.7, decision: "auto" };
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });

    const result = await runCase("case_001", 0.7);

    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining("/cases/case_001/run?tau=0.7"));
    expect(result).toEqual(mockResult);
  });

  it("fetchCases throws when the response is not ok", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => ({}),
    });

    await expect(fetchCases()).rejects.toThrow(/500/);
  });

  it("fetchVolume throws when the response is not ok", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 404,
      json: async () => ({}),
    });

    await expect(fetchVolume("case_001")).rejects.toThrow(/404/);
  });

  it("runCase throws when the response is not ok", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({}),
    });

    await expect(runCase("case_001", 0.7)).rejects.toThrow(/503/);
  });
});
