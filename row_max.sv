module row_max_reduce (
    input  logic signed [15:0] din [16], //16 value input
    output logic signed [15:0] dout      //largest value output
);

    // Level 1: 16 → 8
    logic signed [15:0] l1 [8];
    genvar i;
    generate
        for (i = 0; i < 8; i++) begin : g_l1
            assign l1[i] = (din[2*i] > din[2*i+1]) ? din[2*i] : din[2*i+1]; //cmp 0 and 1...14 and 15
        end
    endgenerate

    // Level 2: 8 → 4
    logic signed [15:0] l2 [4];
    generate
        for (i = 0; i < 4; i++) begin : g_l2
            assign l2[i] = (l1[2*i] > l1[2*i+1]) ? l1[2*i] : l1[2*i+1];  //cmp 0 and 1...7 and 8
        end
    endgenerate

    // Level 3: 4 → 2
    logic signed [15:0] l3 [2];
    assign l3[0] = (l2[0] > l2[1]) ? l2[0] : l2[1];
    assign l3[1] = (l2[2] > l2[3]) ? l2[2] : l2[3];

    // Level 4: 2 → 1
    assign dout = (l3[0] > l3[1]) ? l3[0] : l3[1];

endmodule