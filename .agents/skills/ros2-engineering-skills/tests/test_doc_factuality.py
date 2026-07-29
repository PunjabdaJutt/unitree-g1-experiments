"""Documentation factuality regression tests.

These pin previously discovered factual errors in the reference documents so
they cannot silently reappear. They are string-level regression checks, not
proof of correctness — see README "Current status" for the scope statement.

Background (found during a real Humble/Nav2 field diagnosis):
- navigation.md presented ``recoveries_server`` / ``nav2_recoveries/`` as the
  CURRENT Humble naming; the rename to ``behavior_server`` /
  ``nav2_behaviors/`` actually happened in the Galactic -> Humble migration,
  so the documented config did not exist on the installed Humble.
- The pre-Galactic ``default_bt_xml_filename`` parameter appeared in a config
  example as if current; Galactic+ uses ``default_nav_to_pose_bt_xml`` and
  the old name is silently ignored.
- Spin/BackUp motion recoveries were presented as a mandatory default; an
  unvalidated +1.57 rad Spin caused an unexpected in-place rotation on a
  quadruped in the field.
"""

import os
import re

ROOT = os.path.join(os.path.dirname(__file__), '..')
NAVIGATION_MD = os.path.join(ROOT, 'references', 'navigation.md')
TESTING_MD = os.path.join(ROOT, 'references', 'testing.md')
SKILL_MD = os.path.join(ROOT, 'SKILL.md')
NAV2_EXPECTED = os.path.join(ROOT, 'evals', 'expected',
                             'nav2-configuration.md')

# A line that names a legacy identifier must, on the same line, make clear
# that the identifier is legacy (migration notes, rename explanations).
_LEGACY_CONTEXT = re.compile(
    r'pre-Humble|pre-Galactic|Galactic|legacy|renamed', re.IGNORECASE)


def _read(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return fh.read()


def _lines(path):
    return _read(path).splitlines()


def _md_section(path, heading):
    """Extract heading-to-next-heading from a markdown file, ignoring '#'
    lines inside fenced code blocks (bash comments are not headings)."""
    lines = _lines(path)
    start = next(i for i, line in enumerate(lines)
                 if re.match(r'#{1,6} ', line) and heading in line)
    in_fence = False
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith('```'):
            in_fence = not in_fence
        elif not in_fence and re.match(r'#{1,6} ', lines[i]):
            end = i
            break
    return '\n'.join(lines[start:end])


class TestNav2RecoveryNaming:
    """recoveries_server / nav2_recoveries must never be presented as
    current naming — only inside migration/legacy context."""

    def _assert_legacy_context_only(self, path, needles):
        for lineno, line in enumerate(_lines(path), start=1):
            if any(n in line for n in needles):
                assert _LEGACY_CONTEXT.search(line), (
                    f'{os.path.basename(path)}:{lineno} mentions legacy Nav2 '
                    f'recovery naming outside a migration/legacy context: '
                    f'{line.strip()!r}'
                )

    def test_navigation_md(self):
        self._assert_legacy_context_only(
            NAVIGATION_MD, ('recoveries_server', 'nav2_recoveries'))

    def test_skill_md(self):
        self._assert_legacy_context_only(
            SKILL_MD, ('recoveries_server', 'nav2_recoveries'))

    def test_nav2_expected_has_no_legacy_recovery_naming(self):
        content = _read(NAV2_EXPECTED)
        assert 'recoveries_server' not in content
        assert 'nav2_recoveries' not in content

    def test_navigation_md_documents_current_humble_naming(self):
        content = _read(NAVIGATION_MD)
        assert 'behavior_server:' in content, (
            'navigation.md must show a behavior_server config example')
        assert 'nav2_behaviors/Spin' in content, (
            'navigation.md must document the Humble/Iron plugin syntax')


class TestPluginTypeSeparator:
    """Nav2 plugin type strings use `/` on Humble/Iron and `::` on Jazzy+
    (some types like dwb_core::DWBLocalPlanner already used `::` earlier).
    Both syntaxes must stay documented, and each occurrence must sit in the
    right distro context — string presence alone would pass even if the
    syntaxes were attributed to the wrong distros."""

    _WINDOW = 3

    def _occurrence_windows(self, path, needle):
        lines = _lines(path)
        for idx, line in enumerate(lines):
            if needle in line:
                lo = max(0, idx - self._WINDOW)
                hi = min(len(lines), idx + self._WINDOW + 1)
                yield idx + 1, '\n'.join(lines[lo:hi])

    def test_humble_slash_syntax_in_humble_context(self):
        found = False
        for lineno, window in self._occurrence_windows(
                NAVIGATION_MD, 'nav2_behaviors/Spin'):
            found = True
            assert re.search(r'Humble|Iron|older', window, re.IGNORECASE), (
                f'navigation.md:{lineno} shows the / plugin syntax without '
                f'Humble/Iron context'
            )
        assert found, 'navigation.md must document nav2_behaviors/Spin'

    def test_jazzy_double_colon_syntax_in_jazzy_context(self):
        found = False
        for lineno, window in self._occurrence_windows(
                NAVIGATION_MD, 'nav2_behaviors::Spin'):
            found = True
            assert re.search(r'Jazzy|newer', window, re.IGNORECASE), (
                f'navigation.md:{lineno} shows the :: plugin syntax without '
                f'Jazzy+ context'
            )
        assert found, 'navigation.md must document nav2_behaviors::Spin'

    def test_jazzy_expected_uses_modern_plugin_separator(self):
        """The nav2 eval prompt targets Jazzy — its expected answer must use
        the :: types and carry no /-style behavior plugin strings."""
        content = _read(NAV2_EXPECTED)
        assert 'nav2_behaviors::Spin' in content
        assert 'nav2_behaviors/Spin' not in content


class TestBtNavigatorParameter:
    """default_bt_xml_filename is pre-Galactic (a DIFFERENT boundary from
    the recovery rename) and must only appear flagged as such."""

    def test_old_param_only_in_pre_galactic_context(self):
        marker = re.compile(r'pre-Galactic|Foxy', re.IGNORECASE)
        for path in (NAVIGATION_MD, SKILL_MD, NAV2_EXPECTED):
            for lineno, line in enumerate(_lines(path), start=1):
                if 'default_bt_xml_filename' in line:
                    assert marker.search(line), (
                        f'{os.path.basename(path)}:{lineno} presents the '
                        f'pre-Galactic default_bt_xml_filename without '
                        f'flagging it: {line.strip()!r}'
                    )

    def test_current_param_documented(self):
        for path in (NAVIGATION_MD, NAV2_EXPECTED):
            assert 'default_nav_to_pose_bt_xml' in _read(path), (
                f'{os.path.basename(path)} must document the Galactic+ '
                f'default_nav_to_pose_bt_xml parameter'
            )

    def test_old_default_bt_filename_gone(self):
        """The pre-Galactic default BT filename must not be presented; the
        Humble-era name is navigate_to_pose_w_replanning_and_recovery.xml
        (which does not contain the old token as a substring)."""
        for path in (NAVIGATION_MD, NAV2_EXPECTED):
            assert 'navigate_w_replanning_and_recovery' not in _read(path), (
                f'{os.path.basename(path)} still names the pre-Galactic '
                f'default BT file'
            )


class TestMotionRecoveryGating:
    """Motion recoveries (Spin/BackUp) must not be presented as a mandatory
    default anywhere in the docs or eval fixtures."""

    def test_expected_does_not_mandate_spin_backup(self):
        content = _read(NAV2_EXPECTED)
        assert 'Must configure recovery behaviors: spin' not in content
        assert re.search(r'gate|actuation-free', content), (
            'nav2 expected answer must require safety gating for motion '
            'recoveries'
        )

    def test_navigation_md_keeps_escalation_ladder(self):
        content = _read(NAVIGATION_MD)
        assert 'actuation-free' in content
        assert 'escalation ladder' in content

    def test_costmap_clear_caveat_present(self):
        """Costmap clearing is actuation-free but not automatically safe —
        the re-observation caveat must stay documented."""
        content = _read(NAVIGATION_MD)
        assert re.search(r'can erase \*?real\*? obstacles', content), (
            'navigation.md must keep the costmap-clearing re-observation '
            'caveat'
        )


class TestUseSimTimeExamples:
    """use_sim_time: True examples must always pair with a /clock source
    (ros2 bag play --clock or a simulator); a clockless example hangs
    timer-driven nodes."""

    _WINDOW = 15

    def test_testing_md_sim_time_examples_have_clock_source(self):
        lines = _lines(TESTING_MD)
        for idx, line in enumerate(lines):
            if "'use_sim_time': True" in line:
                lo = max(0, idx - self._WINDOW)
                hi = min(len(lines), idx + self._WINDOW + 1)
                window = '\n'.join(lines[lo:hi])
                assert '--clock' in window, (
                    f'testing.md:{idx + 1} sets use_sim_time without a '
                    f'/clock source nearby'
                )


HARDWARE_MD = os.path.join(ROOT, 'references', 'hardware-interface.md')


class TestDistroMatrix:
    """Row-scoped checks on the SKILL.md distro table.

    Deliberately anchored to specific table cells (not global string
    absence) so that changelog prose mentioning an old value never trips
    the check. Pinned errors: Kilted EOL was recorded as Nov 2025 (official
    EOL is Dec 2026), Kilted ros2_control was 4.x while
    hardware-interface.md said 5.x, and the pre-Lyrical EventsExecutor was
    labeled stable although it ships in rclcpp::experimental.
    """

    def _distro_table(self):
        lines = _lines(SKILL_MD)
        header_idx = next(
            i for i, line in enumerate(lines)
            if line.startswith('| Feature') and 'Humble' in line)
        headers = [c.strip() for c in
                   lines[header_idx].strip().strip('|').split('|')]
        rows = {}
        for line in lines[header_idx + 2:]:
            if not line.startswith('|'):
                break
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            rows[cells[0]] = dict(zip(headers[1:], cells[1:]))
        return rows

    def test_table_has_lyrical_column(self):
        rows = self._distro_table()
        assert 'Lyrical (LTS)' in rows['EOL'], (
            'SKILL.md distro table must carry a Lyrical (LTS) column')

    def test_kilted_eol_cell_is_dec_2026(self):
        assert self._distro_table()['EOL']['Kilted (non-LTS)'] == 'Dec 2026'

    def test_lyrical_eol_cell_is_may_2031(self):
        assert self._distro_table()['EOL']['Lyrical (LTS)'] == 'May 2031'

    def test_lyrical_ubuntu_cell_is_26_04(self):
        assert self._distro_table()['Ubuntu']['Lyrical (LTS)'] == '26.04'

    def test_ros2_control_versions_per_distro(self):
        row = self._distro_table()['ros2_control interface']
        assert row['Kilted (non-LTS)'] == '5.x'
        assert '6.x' in row['Lyrical (LTS)']

    def test_events_executor_cells(self):
        row = self._distro_table()['EventsExecutor']
        assert 'Experimental' in row['Kilted (non-LTS)']
        assert 'EventsCBGExecutor' in row['Lyrical (LTS)']

    def test_hardware_interface_agrees_on_kilted_5x(self):
        content = _read(HARDWARE_MD)
        assert re.search(r'Kilted \(5\.x\)', content), (
            'hardware-interface.md distro comparison must keep Kilted at '
            '5.x, matching the SKILL.md table')


class TestCallbackGroupRule:
    """Service-from-callback semantics: synchronous waiting deadlocks a
    MutuallyExclusiveCallbackGroup; registering a response callback and
    returning is safe even in the same group. Pinned error: an anti-pattern
    row claimed the pattern 'Deadlocks even with async', prescribing group
    surgery where none was needed."""

    def _skill_table_row(self, needle):
        for line in _lines(SKILL_MD):
            if line.startswith('|') and needle in line:
                return line
        raise AssertionError(
            f'no SKILL.md table row containing {needle!r}')

    def test_service_future_row_states_async_is_safe(self):
        row = self._skill_table_row('service future')
        assert 'safe even in the same group' in row
        assert 'Deadlocks even with async' not in row

    def test_principle_bullet_distinguishes_sync_wait(self):
        lines = _lines(SKILL_MD)
        start = next(i for i, line in enumerate(lines)
                     if 'Calling a service from a callback' in line)
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].startswith('- ') or not lines[i].strip()),
                   len(lines))
        block = '\n'.join(lines[start:end])
        assert 'returns without waiting' in block
        assert 'add_done_callback' in block
        assert 'must** be in a separate' not in block


COMMUNICATION_MD = os.path.join(ROOT, 'references', 'communication.md')


class TestDefaultRmwVendor:
    """Fast DDS (rmw_fastrtps_cpp) is the default RMW on every current
    release; only Galactic defaulted to CycloneDDS. Pinned errors: a
    section heading called CycloneDDS the 'default vendor' and the vendor
    table credited CycloneDDS as default for Humble+."""

    def test_cyclonedds_heading_not_called_default(self):
        for lineno, line in enumerate(_lines(COMMUNICATION_MD), start=1):
            if line.startswith('#') and 'CycloneDDS tuning' in line:
                assert 'default vendor' not in line, (
                    f'communication.md:{lineno} still titles CycloneDDS as '
                    f'the default vendor'
                )
                return
        raise AssertionError('CycloneDDS tuning section heading not found')

    def test_vendor_table_default_row(self):
        row = next((line for line in _lines(COMMUNICATION_MD)
                    if line.startswith('|') and 'Default RMW' in line),
                   None)
        assert row is not None, 'vendor table must keep a Default RMW row'
        cells = [c.strip() for c in row.strip().strip('|').split('|')]
        # Column order: Aspect | CycloneDDS | FastDDS | Connext | Zenoh
        assert 'Galactic' in cells[1], (
            f'CycloneDDS default cell must say Galactic only: {cells[1]!r}')
        assert 'Default' in cells[2], (
            f'FastDDS cell must carry the Default marker: {cells[2]!r}')

    def test_vendor_table_separates_support_tier(self):
        """Default-RMW status and ROS support tier are different facts and
        must live in separate rows (the old combined row put Connext's
        Tier 2 in the default column)."""
        assert any(line.startswith('|') and 'ROS support tier' in line
                   for line in _lines(COMMUNICATION_MD))

    def test_default_rmw_stated_in_dds_section(self):
        content = _read(COMMUNICATION_MD)
        assert 'rmw_fastrtps_cpp' in content


class TestZeroCopyClaims:
    """Copy avoidance is conditional, split across three mechanisms
    (rclcpp intra-process, loaned messages/SHM, separate-process DDS).
    Pinned error: SKILL.md called separate-process intra-host DDS
    'zero-overhead' and presented intra-process transfer as unconditionally
    zero-copy."""

    def test_skill_md_only_negates_zero_overhead(self):
        """'zero-overhead' may appear only in its negation."""
        for lineno, line in enumerate(_lines(SKILL_MD), start=1):
            plain = line.replace('**', '')
            idx = 0
            while True:
                idx = plain.find('zero-overhead', idx)
                if idx == -1:
                    break
                assert plain[max(0, idx - 4):idx] == 'not ', (
                    f'SKILL.md:{lineno} asserts zero-overhead positively: '
                    f'{line.strip()!r}'
                )
                idx += 1

    def test_skill_md_states_dds_is_not_zero_overhead(self):
        assert 'not zero-overhead' in _read(SKILL_MD).replace('**', '')

    def test_nodes_executors_lists_three_mechanisms(self):
        content = _read(os.path.join(ROOT, 'references',
                                     'nodes-executors.md'))
        assert 'Loaned messages' in content
        assert 'standard DDS transport' in content
        assert 'subscriber count' in content.lower()

    def test_interprocess_copy_avoidance_not_absolutized(self):
        """Vendor SHM/PSMX and loaned messages CAN avoid copies across
        processes when their preconditions hold — the docs must not claim
        inter-process is 'never' zero-copy (pinned overcorrection)."""
        for path in (SKILL_MD,
                     os.path.join(ROOT, 'references', 'nodes-executors.md')):
            content = _read(path)
            assert 'never zero-overhead' not in content, path
            assert 'never zero-copy' not in content, path


class TestCrashSafetyFraming:
    """Destructors are best-effort cleanup, not crash safety: they never
    run on SIGKILL/power loss and are not guaranteed on segfaults. Crash
    safety lives downstream (command timeout, heartbeat/watchdog, hardware
    e-stop). Pinned error: pitfall 8 and the shutdown anti-pattern row
    presented on_deactivate + destructor as the fix for crashes, and
    hardware-interface.md called the destructor a crash safety net."""

    def _skill_row(self, needle):
        for line in _lines(SKILL_MD):
            if line.startswith('|') and needle in line:
                return line
        raise AssertionError(f'no SKILL.md table row containing {needle!r}')

    def test_pitfall_8_requires_downstream_failsafes(self):
        row = self._skill_row('Treating process-side cleanup')
        assert 'best-effort' in row
        assert re.search(r'watchdog|timeout', row)
        assert 'when the node crashes | Send zero-command' not in row

    def test_shutdown_antipattern_row_points_downstream(self):
        row = self._skill_row('No safe command on shutdown')
        assert 'best-effort' in row
        assert re.search(r'watchdog|timeout', row)
        assert 'safety-estop' in row

    def test_hardware_interface_destructor_not_a_safety_net(self):
        content = _read(HARDWARE_MD)
        assert 'NOT a crash safety net' in content
        assert 'destructor does not run' in content
        assert 'not guaranteed' in content
        assert 'safety-estop' in content


class TestBlockingWaitSemantics:
    """Blocking-wait examples must be split by client library. Pinned
    error: rclpy's Future.result() was listed as a synchronous wait — it
    does not block; it immediately returns whatever result is stored."""

    def _callback_bullet(self):
        lines = _lines(SKILL_MD)
        start = next(i for i, line in enumerate(lines)
                     if 'Calling a service from a callback' in line)
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].startswith('- ') or not lines[i].strip()),
                   len(lines))
        # Normalize hard-wrapped prose so phrase assertions are not broken
        # by line-wrap positions.
        return ' '.join('\n'.join(lines[start:end]).split())

    def test_rclpy_future_result_not_listed_as_blocking(self):
        block = self._callback_bullet()
        assert 'does not block' in block, (
            'the bullet must state that rclpy future.result() does not '
            'block')
        assert 'future.done()' in block

    def test_rclcpp_get_qualified_to_incomplete_futures(self):
        """future.get() is only a deadlock when called on a not-yet-complete
        future from the initiating callback; inside the response callback
        the future is complete and get() is safe (the examples use it)."""
        block = self._callback_bullet()
        assert 'not-yet-complete' in block
        assert '`get()`' in block
        assert 'response callback' in block


class TestKillTestSafetyConditions:
    """The kill -9 failsafe verification moves a real robot if the
    failsafe is broken; the instruction must carry its safety
    preconditions inline, not in another file."""

    def test_kill_test_paragraph_carries_safety_conditions(self):
        lines = _lines(HARDWARE_MD)
        idx = next(i for i, line in enumerate(lines) if 'kill -9' in line)
        window = '\n'.join(lines[max(0, idx - 3):idx + 10])
        for needle in ('never an automated', 'simulation', 'e-stop',
                       'torque limits'):
            assert needle in window, (
                f'kill-test instructions missing inline safety condition: '
                f'{needle!r}'
            )


SAFETY_ESTOP_MD = os.path.join(ROOT, 'references', 'safety-estop.md')


class TestFaultInjectionSafetyConsistency:
    """Fault injection and spoof tests on a real robot are operator-approved
    tests on a physically restrained platform, never unattended CI or an
    agent action. Pinned errors: the stop-path checklist said to automate
    items 1-6 'in CI-on-robot', and the SROS2 isolation check said 'Test
    this as part of CI-on-robot' — both contradicting hardware-interface.md.

    Scoped to the two sections that instruct hardware tests (heading to
    next heading), not a global string ban: conditional mentions of
    CI-on-robot elsewhere (e.g. 'do NOT run this in CI-on-robot') must stay
    legal."""

    def _section(self, heading):
        return _md_section(SAFETY_ESTOP_MD, heading)

    def test_isolation_check_requires_safety_conditions(self):
        section = self._section('Verify the isolation')
        assert 'Test this as part of CI-on-robot' not in section
        for needle in ('simulation/HIL', 'operator-approved', 'restrained',
                       'unattended CI', 'AI agent'):
            assert needle in section, (
                f'isolation-check section missing safety condition: '
                f'{needle!r}'
            )

    def test_stop_path_checklist_requires_safety_conditions(self):
        section = self._section('Stop-path checklist')
        assert 'Test this as part of CI-on-robot' not in section
        for needle in ('operator approval', 'physically restrained',
                       'unattended CI', 'AI agent'):
            assert needle in section, (
                f'stop-path checklist section missing safety condition: '
                f'{needle!r}'
            )


REALTIME_MD = os.path.join(ROOT, 'references', 'realtime.md')


class TestCyclictestProcedureSingleSource:
    """realtime.md defines ONE cyclictest measurement procedure (in the
    Benchmark reference numbers section); the tail-latency section
    references it instead of carrying a second recipe. Pinned error: the
    tail-latency section kept an older command (-l 100000, -h 400 — a
    histogram ceiling that overflows non-RT spikes) and a 'sort histogram'
    analysis that conflicted with the cumulative-count readout."""

    def test_tail_latency_section_references_the_procedure(self):
        section = _md_section(REALTIME_MD, 'Tail latency')
        assert '(#benchmark-reference-numbers)' in section, (
            'tail-latency section must link to the single measurement '
            'procedure')
        for stale in ('sort histogram', '-h 400', '-l 100000',
                      'sudo cyclictest'):
            assert stale not in section, (
                f'tail-latency section must not carry its own recipe: '
                f'{stale!r}'
            )

    def test_benchmark_section_keeps_the_procedure(self):
        section = _md_section(REALTIME_MD, 'Benchmark reference numbers')
        for needle in ('-D 10m', '-h 20000', 'cumulative'):
            assert needle in section, (
                f'benchmark procedure missing element: {needle!r}')
        assert re.search(r'overflows?', section), (
            'benchmark procedure must keep the overflow rerun note')
