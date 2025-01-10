// styles/dashboardStyles.js
export const dashboardStyles = {
  mainContainer: "min-h-screen bg-[#001F3F]",
  contentContainer: "max-w-7xl mx-auto px-8 py-6",

  // Cards
  cardGrid: "grid grid-cols-1 md:grid-cols-3 gap-4 mb-8",
  card: "bg-[#002851] rounded-lg shadow-sm p-6 border border-[#003366]",
  cardTitle: "text-sm font-medium text-gray-400",
  cardValue: "text-2xl font-bold text-[#87CEEB] mt-2",

  // Stats
  statsContainer: "flex justify-between items-start",
  iconContainer: "h-10 w-10 rounded-lg border border-[#003366] flex items-center justify-center",
  icon: "h-6 w-6 text-[#87CEEB]",

  // Chart
  chartCard: "bg-[#002851] rounded-lg shadow-sm p-6 border border-gray-200",
  chartTitle: "text-lg font-semibold text-[#87CEEB]",

  // Table
  tableContainer: "bg-[#002851] rounded-lg shadow-sm border border-gray-200",
  tableHeader: "px-6 py-3 border border-gray-200",
  tableCell: "text-lg font-semibold text-gray-900",

  // Status badge
  statusBadge: {
    active: "px-3 py-1 rounded-full border bg-green-100 text-green-800 border-green-200",
    inactive: "px-3 py-1 rounded-full border bg-gray-100 text-gray-800 border-gray-200"
  },

  // Buttons
  button: "px-4 py-2 rounded-lg transition-colors duration-200",
  primaryButton: "bg-[#87CEEB] hover:bg-[#5F9EA0] text-white",
  stopButton: "bg-red-500 hover:bg-red-600 text-white"
};
