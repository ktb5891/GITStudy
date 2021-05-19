window.addEventListener("load", function () {
  // DataTable.js
  //Code for months to apply for sorting order
  var monthNames = {};
  monthNames["January"] = "01";
  monthNames["February"] = "02";
  monthNames["March"] = "03";
  monthNames["April"] = "04";
  monthNames["May"] = "05";
  monthNames["June"] = "06";
  monthNames["July"] = "07";
  monthNames["August"] = "08";
  monthNames["September"] = "09";
  monthNames["October"] = "10";
  monthNames["November"] = "11";
  monthNames["December"] = "12";

  //Code to help with Sorting the Cert Number in proper order
  jQuery.fn.dataTableExt.oSort["numeric-comma-asc"] = function (a, b) {
    var x = a == "-" ? 0 : a.replace(/,/, ".");
    var y = b == "-" ? 0 : b.replace(/,/, ".");
    x = parseFloat(x);
    y = parseFloat(y);
    return x < y ? -1 : x > y ? 1 : 0;
  };

  jQuery.fn.dataTableExt.oSort["numeric-comma-desc"] = function (a, b) {
    var x = a == "-" ? 0 : a.replace(/,/, ".");
    var y = b == "-" ? 0 : b.replace(/,/, ".");
    x = parseFloat(x);
    y = parseFloat(y);
    return x < y ? 1 : x > y ? -1 : 0;
  };

  // Code for Table display and function
  // NOTE: this code is unique to work with the data provided a
  var datatableOptions = {
    // data: datatableData,
    // columns: [
    //   { data: "url" },
    //   { data: "imageSrc" },
    //   { data: "type" },
    //   { data: "date" },
    //   NOTE: we only render one table row cell in order to build our card
    //   { data: "content" }
    // ],
    autoWidth: false,
    // fixedColumns: true,
    sDom:
      "<'position-relative' <'data-table--header usa-form position-sticky top-68px tablet:top-94px z-200' iflp> <'data-table--content' rt> <'data-table--footer' p>>",
    pagingType: "first_last_numbers",
    pageLength: 12,
    order: [],
    language: {
      infoFiltered: "",
      search: "_INPUT_",
      searchPlaceholder: "Search this Listing",
      lengthMenu:
        '<select class="usa-select margin-top-0" aria-label="Results displayed">' +
        '<option value="12">12</option>' +
        '<option value="24">24</option>' +
        '<option value="60">60</option>' +
        '<option value="100">100</option>' +
        '<option value="-1">All</option>' +
        "</select>",
    },
    initComplete: function (settings, json) {
      $(".data-table--header.usa-form input").addClass(
        "usa-input usa-input-search"
      );

      if (navigator.userAgent.match(/Trident\/7\./)) {
        //need offset due to menu
        $(".data-table--header.position-sticky").stickybits({
          useStickyClasses: true,
        });
      }
    },
  };
  $(".dataTable.dataTable--cards").dataTable(datatableOptions);
  var datatableOptionsForSidebar = Object.assign({}, datatableOptions);
  (datatableOptionsForSidebar.columns = [
	  { type: "html", targets: 0 }, // Bank Name
	  null, // City
	  null, // State
	  null, // Cert
	  null, // Acquiring Institution
	  { type: "date", targets: 0 }, // closing date
	  null, // fund number
  ]),
	  
  $(".dataTable.dataTables-sidebar").dataTable(datatableOptionsForSidebar);
});
