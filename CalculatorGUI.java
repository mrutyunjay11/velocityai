import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.*;
import java.text.DecimalFormat;

/**
 * Modern Java GUI Calculator
 * Features:
 * - Clean modern dark UI
 * - Basic operations (+, -, *, /, %)
 * - Scientific/utility operations (sqrt, x^2, 1/x, +/-)
 * - Clear (C), Clear Entry (CE), Backspace
 * - Keyboard navigation and typing support
 * - Clean number formatting without trailing zeros
 */
public class CalculatorGUI extends JFrame implements ActionListener, KeyListener {

    private final JTextField display;
    private final JLabel historyLabel;

    private double firstOperand = 0;
    private String pendingOperator = "";
    private boolean startNewNumber = true;
    private final DecimalFormat df = new DecimalFormat("#.##########");

    // Colors
    private final Color BG_COLOR = new Color(28, 28, 30);
    private final Color DISPLAY_BG = new Color(44, 44, 46);
    private final Color BTN_NUM_BG = new Color(58, 58, 60);
    private final Color BTN_OP_BG = new Color(255, 159, 10);
    private final Color BTN_UTIL_BG = new Color(99, 99, 102);
    private final Color TEXT_COLOR = Color.WHITE;

    public CalculatorGUI() {
        setTitle("VelocityAI Calculator");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(380, 520);
        setLocationRelativeTo(null);
        setResizable(false);
        getContentPane().setBackground(BG_COLOR);
        setLayout(new BorderLayout(10, 10));

        // Top Panel: History and Main Display
        JPanel topPanel = new JPanel();
        topPanel.setLayout(new BoxLayout(topPanel, BoxLayout.Y_AXIS));
        topPanel.setBackground(BG_COLOR);
        topPanel.setBorder(new EmptyBorder(15, 15, 5, 15));

        historyLabel = new JLabel(" ");
        historyLabel.setFont(new Font("SansSerif", Font.PLAIN, 14));
        historyLabel.setForeground(new Color(174, 174, 178));
        historyLabel.setAlignmentX(Component.RIGHT_ALIGNMENT);
        topPanel.add(historyLabel);

        display = new JTextField("0");
        display.setFont(new Font("SansSerif", Font.BOLD, 36));
        display.setHorizontalAlignment(JTextField.RIGHT);
        display.setEditable(false);
        display.setBackground(DISPLAY_BG);
        display.setForeground(TEXT_COLOR);
        display.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(new Color(72, 72, 74), 1),
            new EmptyBorder(10, 15, 10, 15)
        ));
        display.addKeyListener(this);
        topPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        topPanel.add(display);

        add(topPanel, BorderLayout.NORTH);

        // Buttons Grid
        JPanel buttonsPanel = new JPanel();
        buttonsPanel.setBackground(BG_COLOR);
        buttonsPanel.setBorder(new EmptyBorder(5, 15, 15, 15));
        buttonsPanel.setLayout(new GridLayout(6, 4, 8, 8));

        String[] buttonLabels = {
            "%", "CE", "C", "⌫",
            "1/x", "x²", "√x", "÷",
            "7", "8", "9", "×",
            "4", "5", "6", "-",
            "1", "2", "3", "+",
            "±", "0", ".", "="
        };

        for (String label : buttonLabels) {
            JButton btn = createButton(label);
            buttonsPanel.add(btn);
        }

        add(buttonsPanel, BorderLayout.CENTER);
        addKeyListener(this);
        setFocusable(true);
    }

    private JButton createButton(String text) {
        JButton btn = new JButton(text);
        btn.setFont(new Font("SansSerif", Font.BOLD, 18));
        btn.setFocusPainted(false);
        btn.setBorderPainted(false);
        btn.setOpaque(true);
        btn.setForeground(TEXT_COLOR);

        // Styling based on button role
        if (text.matches("[0-9]")) {
            btn.setBackground(BTN_NUM_BG);
        } else if (text.equals("=") || text.equals("+") || text.equals("-") || text.equals("×") || text.equals("÷")) {
            btn.setBackground(BTN_OP_BG);
        } else {
            btn.setBackground(BTN_UTIL_BG);
        }

        btn.addActionListener(this);
        return btn;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        String cmd = e.getActionCommand();
        processCommand(cmd);
    }

    private void processCommand(String cmd) {
        try {
            if (cmd.matches("[0-9]")) {
                if (startNewNumber || display.getText().equals("0")) {
                    display.setText(cmd);
                    startNewNumber = false;
                } else {
                    display.setText(display.getText() + cmd);
                }
            } else if (cmd.equals(".")) {
                if (startNewNumber) {
                    display.setText("0.");
                    startNewNumber = false;
                } else if (!display.getText().contains(".")) {
                    display.setText(display.getText() + ".");
                }
            } else if (cmd.equals("C")) {
                display.setText("0");
                historyLabel.setText(" ");
                firstOperand = 0;
                pendingOperator = "";
                startNewNumber = true;
            } else if (cmd.equals("CE")) {
                display.setText("0");
                startNewNumber = true;
            } else if (cmd.equals("⌫")) {
                String text = display.getText();
                if (!startNewNumber && text.length() > 0) {
                    text = text.substring(0, text.length() - 1);
                    if (text.isEmpty() || text.equals("-")) text = "0";
                    display.setText(text);
                }
            } else if (cmd.equals("±")) {
                double val = Double.parseDouble(display.getText());
                display.setText(formatResult(-val));
            } else if (cmd.equals("√x")) {
                double val = Double.parseDouble(display.getText());
                if (val < 0) {
                    display.setText("Invalid Input");
                    startNewNumber = true;
                } else {
                    historyLabel.setText("√(" + formatResult(val) + ")");
                    display.setText(formatResult(Math.sqrt(val)));
                    startNewNumber = true;
                }
            } else if (cmd.equals("x²")) {
                double val = Double.parseDouble(display.getText());
                historyLabel.setText("sqr(" + formatResult(val) + ")");
                display.setText(formatResult(val * val));
                startNewNumber = true;
            } else if (cmd.equals("1/x")) {
                double val = Double.parseDouble(display.getText());
                if (val == 0) {
                    display.setText("Cannot divide by 0");
                    startNewNumber = true;
                } else {
                    historyLabel.setText("1/(" + formatResult(val) + ")");
                    display.setText(formatResult(1.0 / val));
                    startNewNumber = true;
                }
            } else if (cmd.equals("%")) {
                double val = Double.parseDouble(display.getText());
                display.setText(formatResult(val / 100.0));
                startNewNumber = true;
            } else if (cmd.equals("+") || cmd.equals("-") || cmd.equals("×") || cmd.equals("÷")) {
                if (!pendingOperator.isEmpty() && !startNewNumber) {
                    calculate();
                }
                firstOperand = Double.parseDouble(display.getText());
                pendingOperator = cmd;
                historyLabel.setText(formatResult(firstOperand) + " " + pendingOperator);
                startNewNumber = true;
            } else if (cmd.equals("=")) {
                if (!pendingOperator.isEmpty()) {
                    double second = Double.parseDouble(display.getText());
                    historyLabel.setText(formatResult(firstOperand) + " " + pendingOperator + " " + formatResult(second) + " =");
                    calculate();
                    pendingOperator = "";
                    startNewNumber = true;
                }
            }
        } catch (NumberFormatException ex) {
            display.setText("Error");
            startNewNumber = true;
        }
    }

    private void calculate() {
        double secondOperand = Double.parseDouble(display.getText());
        double res = 0;
        switch (pendingOperator) {
            case "+": res = firstOperand + secondOperand; break;
            case "-": res = firstOperand - secondOperand; break;
            case "×": res = firstOperand * secondOperand; break;
            case "÷":
                if (secondOperand == 0) {
                    display.setText("Cannot divide by 0");
                    return;
                }
                res = firstOperand / secondOperand;
                break;
        }
        display.setText(formatResult(res));
        firstOperand = res;
    }

    private String formatResult(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return "Error";
        }
        return df.format(value);
    }

    // Keyboard support
    @Override
    public void keyTyped(KeyEvent e) {
        char key = e.getKeyChar();
        if (Character.isDigit(key)) {
            processCommand(String.valueOf(key));
        } else if (key == '.') {
            processCommand(".");
        } else if (key == '+') {
            processCommand("+");
        } else if (key == '-') {
            processCommand("-");
        } else if (key == '*' || key == 'x' || key == 'X') {
            processCommand("×");
        } else if (key == '/') {
            processCommand("÷");
        } else if (key == '=' || key == '\n') {
            processCommand("=");
        } else if (key == KeyEvent.VK_BACK_SPACE) {
            processCommand("⌫");
        } else if (key == KeyEvent.VK_ESCAPE) {
            processCommand("C");
        }
    }

    @Override public void keyPressed(KeyEvent e) {}
    @Override public void keyReleased(KeyEvent e) {}

    public static void main(String[] args) {
        // Set system look and feel if available
        try {
            UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeelClassName());
        } catch (Exception ignored) {}

        SwingUtilities.invokeLater(() -> {
            new CalculatorGUI().setVisible(true);
        });
    }
}
